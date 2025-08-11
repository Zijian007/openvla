"""
DPO training utilities and loss computation functions.
"""

import os
import datetime
from collections import deque
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def compute_log_probs(model, prompt_input_ids, completion_input_ids, prompt_attention_mask, completion_attention_mask, model_device=None):
    """计算模型对completion部分的log概率"""
    # 确保输入数据在正确的device上
    if model_device is None:
        model_device = next(model.parameters()).device
    
    prompt_input_ids = prompt_input_ids.to(model_device)
    completion_input_ids = completion_input_ids.to(model_device)
    prompt_attention_mask = prompt_attention_mask.to(model_device)
    completion_attention_mask = completion_attention_mask.to(model_device)
    
    # 拼接prompt和completion
    input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    
    # 前向传播
    with torch.no_grad() if model.training == False else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
    
    # 计算completion部分的log概率
    prompt_len = prompt_input_ids.shape[1]
    completion_logits = logits[:, prompt_len-1:-1, :]  # 获取completion部分的logits
    completion_labels = completion_input_ids
    
    # 计算每个token的log概率
    log_probs = F.log_softmax(completion_logits, dim=-1)
    # 获取对应label的log概率
    selected_log_probs = torch.gather(log_probs, dim=-1, index=completion_labels.unsqueeze(-1)).squeeze(-1)
    # 对非padding的token求和
    sequence_log_probs = (selected_log_probs * completion_attention_mask).sum(dim=-1)
    
    return sequence_log_probs


def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta=0.1, target_device=None):
    """计算DPO损失，处理不同device上的tensor"""
    
    # 确定目标device (通常是policy model的device)
    if target_device is None:
        target_device = policy_chosen_logps.device
    
    # 将所有tensor移动到目标device
    policy_chosen_logps = policy_chosen_logps.to(target_device)
    policy_rejected_logps = policy_rejected_logps.to(target_device)
    ref_chosen_logps = ref_chosen_logps.to(target_device)
    ref_rejected_logps = ref_rejected_logps.to(target_device)
    
    # 计算log比率
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    # DPO损失: -log(sigmoid(beta * (chosen_logratios - rejected_logratios)))
    logits = chosen_logratios - rejected_logratios
    losses = -F.logsigmoid(beta * logits)
    
    # 计算奖励（用于监控）
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    
    return losses.mean(), chosen_rewards, rejected_rewards


def move_batch_to_device(batch, device):
    """将batch中的所有tensor移动到指定device"""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def train_dpo(model, ref_model, train_dataloader, cfg, if_not_demo=False):
    """DPO训练主循环，支持不同device上的模型"""

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    
    exp_id += f"--{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    run_dir = os.path.join(cfg.run_root_dir, exp_id)
    adapter_dir = os.path.join(cfg.adapter_tmp_dir, exp_id)
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    
    if if_not_demo:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"DPO+{exp_id}")
    
    # 获取模型所在的device
    policy_device = next(model.parameters()).device
    ref_device = next(ref_model.parameters()).device
    
    print(f"Policy model device: {policy_device}")
    print(f"Reference model device: {ref_device}")
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    
    # 确保参考模型不更新
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rewards_margin = deque(maxlen=cfg.grad_accumulation_steps)
    recent_chosen_logps = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rejected_logps = deque(maxlen=cfg.grad_accumulation_steps)
    recent_chosen_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rejected_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    
    # 训练循环
    with tqdm(total=cfg.max_steps, leave=False) as progress:
        model.train()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Check if we've reached max steps
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            if gradient_step_idx >= cfg.max_steps:
                break
                
            # 将batch移动到policy model的device
            policy_batch = move_batch_to_device(batch, policy_device)
            
            # 1. 计算策略模型的log概率 (在policy_device上)
            policy_chosen_logps = compute_log_probs(
                model, 
                policy_batch['prompt_input_ids'], 
                policy_batch['chosen_input_ids'], 
                policy_batch['prompt_attention_mask'], 
                policy_batch['chosen_attention_mask'],
                model_device=policy_device
            )
            policy_rejected_logps = compute_log_probs(
                model, 
                policy_batch['prompt_input_ids'], 
                policy_batch['rejected_input_ids'],
                policy_batch['prompt_attention_mask'], 
                policy_batch['rejected_attention_mask'],
                model_device=policy_device
            )
            
            # 2. 计算参考模型的log概率 (在ref_device上)
            with torch.no_grad():
                # 将batch移动到ref model的device
                ref_batch = move_batch_to_device(batch, ref_device)
                
                ref_chosen_logps = compute_log_probs(
                    ref_model, 
                    ref_batch['prompt_input_ids'], 
                    ref_batch['chosen_input_ids'],
                    ref_batch['prompt_attention_mask'], 
                    ref_batch['chosen_attention_mask'],
                    model_device=ref_device
                )
                ref_rejected_logps = compute_log_probs(
                    ref_model, 
                    ref_batch['prompt_input_ids'], 
                    ref_batch['rejected_input_ids'],
                    ref_batch['prompt_attention_mask'], 
                    ref_batch['rejected_attention_mask'],
                    model_device=ref_device
                )
            
            # 3. 计算DPO损失 (在policy_device上)
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=cfg.dpo_beta, target_device=policy_device
            )
            
            # 4. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress.update()

            # 5. 计算指标
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()

            recent_losses.append(loss.item())
            recent_accuracies.append(accuracy.item())
            recent_rewards_margin.append(reward_margin.item())
            recent_chosen_logps.append(policy_chosen_logps.mean().item())
            recent_rejected_logps.append(policy_rejected_logps.mean().item())
            recent_chosen_rewards.append(chosen_rewards.mean().item())
            recent_rejected_rewards.append(rejected_rewards.mean().item())

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_accuracy = sum(recent_accuracies) / len(recent_accuracies)
            smoothened_rewards_margin = sum(recent_rewards_margin) / len(recent_rewards_margin)
            smoothened_chosen_logps = sum(recent_chosen_logps) / len(recent_chosen_logps)
            smoothened_rejected_logps = sum(recent_rejected_logps) / len(recent_rejected_logps)
            smoothened_chosen_rewards = sum(recent_chosen_rewards) / len(recent_chosen_rewards)
            smoothened_rejected_rewards = sum(recent_rejected_rewards) / len(recent_rejected_rewards)

            if gradient_step_idx % 1 == 0:
                if if_not_demo:
                    wandb.log({
                        "loss": smoothened_loss,
                        "accuracy": smoothened_accuracy,
                        "reward_margin": smoothened_rewards_margin,
                        "chosen_logps": smoothened_chosen_logps,
                        "rejected_logps": smoothened_rejected_logps,
                        "chosen_rewards": smoothened_chosen_rewards,
                        "rejected_rewards": smoothened_rejected_rewards
                    },
                    step=gradient_step_idx,
                    )
            
            if gradient_step_idx % 10 == 0:
                model.save_pretrained(adapter_dir)
                print(f"Saved adapter to {adapter_dir}, batch_idx: {batch_idx}")
                
            # 定期清理缓存（如果使用GPU）
            if gradient_step_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    print(f"Training completed! Final adapter saved to: {adapter_dir}")
    return adapter_dir
