"""
DPO training utilities and loss computation functions.
"""

import os
import datetime
import itertools
from collections import deque
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def compute_log_probs(model, prompt_input_ids, completion_input_ids, prompt_attention_mask, completion_attention_mask, 
                      is_chosen=True, is_policy=True, model_device=None):
    """计算模型对completion部分的log概率
    
    Args:
        model: 要计算的模型
        prompt_input_ids: prompt部分的token ids
        completion_input_ids: completion部分的token ids
        prompt_attention_mask: prompt部分的attention mask
        completion_attention_mask: completion部分的attention mask
        is_chosen: bool, True表示计算chosen completion, False表示rejected completion
        is_policy: bool, True表示policy model, False表示reference model
        model_device: 模型所在的device
    
    Returns:
        tuple: (sequence_log_probs, is_valid)
               sequence_log_probs: log概率 (如果is_valid=False则为None)
               is_valid: bool, 表示数据是否有效
    """
    # 确保输入数据在正确的device上
    if model_device is None:
        model_device = next(model.parameters()).device
    
    prompt_input_ids = prompt_input_ids.to(model_device)  # [batch_size, prompt_len]
    completion_input_ids = completion_input_ids.to(model_device)  # [batch_size, completion_len]
    prompt_attention_mask = prompt_attention_mask.to(model_device)  # [batch_size, prompt_len]
    completion_attention_mask = completion_attention_mask.to(model_device)  # [batch_size, completion_len]

    # 验证completion_input_ids的结构：长度能被8整除，且每8个token为一组时，每组最后一个token是32001
    batch_size, completion_len = completion_input_ids.shape
    if completion_len % 8 != 0:
        print(f"     Warning: completion_input_ids length {completion_len} is not divisible by 8, skipping batch")
        return None, None, False
    
    # 检查每组最后一个token是否为32001
    reshaped_completion = completion_input_ids.view(batch_size, completion_len // 8, 8)
    last_tokens = reshaped_completion[:, :, 7]  # 每组的最后一个token
    if not torch.all(last_tokens == 32001):
        print(f"     Warning: Not all groups end with token 32001, got {last_tokens.tolist()}, skipping batch")
        return None, None, False
    
    # 拼接prompt和completion
    input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    
    # 前向传播
    with torch.no_grad() if model.training == False else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

    # 计算completion部分的log概率
    prompt_len = prompt_input_ids.shape[1]
    completion_logits = logits[:, prompt_len:, :]  # 获取completion部分的logits
    completion_labels = completion_input_ids

    # 只对policy chosen执行预测序列比较
    if is_policy and is_chosen:
        # 计算当前的组数
        num_groups = completion_len // 8
        print(f"Current action stream length: {num_groups}")
        completion_logits = logits[:, prompt_len:, :]  # 获取completion部分的logits
        predicted_tokens = torch.argmax(logits[:, prompt_len-1:-1, :], dim = -1)  # [batch_size, completion_len]
        
        # 找到第一个不匹配的token位置
        if predicted_tokens.shape[0] > 0:
            first_pred = predicted_tokens[0]
            first_true = completion_input_ids[0]
            diff_mask = (first_pred != first_true)
            if torch.any(diff_mask):
                first_diff_pos = torch.where(diff_mask)[0][0].item()
                print(f"     Policy chosen prediction differs from true at token position: {first_diff_pos}")
            else:
                print("      Policy chosen prediction matches true completion perfectly")
    
    # 计算每个token的log概率
    log_probs = F.log_softmax(completion_logits, dim=-1) # [batch_size, seq_len, vocab_size]
    # 获取对应label的log概率
    selected_log_probs = torch.gather(log_probs, dim=-1, index=completion_labels.unsqueeze(-1)).squeeze(-1) # [batch_size, seq_len]

    # 修改completion_attention_mask: 每8个token为一组，忽略每组第8个token
    batch_size, completion_len = completion_attention_mask.shape
    assert completion_len % 8 == 0, f"completion_attention_mask length {completion_len} must be divisible by 8 in training_utils.py/compute_log_probs"
    # 将attention_mask reshape为 [batch_size, num_groups, 8]
    num_groups = completion_len // 8
    reshaped_mask = completion_attention_mask.view(batch_size, num_groups, 8)
    # 将每组的第8个token的mask设为0（忽略）
    reshaped_mask[:, :, 7] = 0
    # 重新flatten回原来的形状
    completion_attention_mask = reshaped_mask.view(batch_size, completion_len)
    # 对非padding的token求和
    sequence_log_probs = (selected_log_probs * completion_attention_mask).sum(dim=-1) #torch.Size([1])
    
    # 计算每个组各自的和
    masked_log_probs = selected_log_probs * completion_attention_mask  # [batch_size, completion_len]
    # 将masked_log_probs reshape为 [batch_size, num_groups, 8]
    group_log_probs = masked_log_probs.view(batch_size, num_groups, 8)
    # 对每个组求和，得到 [batch_size, num_groups]
    grouped_log_probs = group_log_probs.sum(dim=-1)

    # 验证grouped_log_probs求和是否等于sequence_log_probs
    grouped_sum = grouped_log_probs.sum(dim=-1)  # [batch_size]
    
    # 检查数值是否相等（使用小的容差处理浮点数精度问题）
    tolerance = 1e-6
    is_equal = torch.allclose(sequence_log_probs, grouped_sum, atol=tolerance)
    
    if not is_equal:
        print(f"Warning: grouped_log_probs sum ({grouped_sum.item()}) does not equal sequence_log_probs ({sequence_log_probs.item()})")
        # print(f"Difference: {torch.abs(sequence_log_probs - grouped_sum)}")
    return sequence_log_probs, grouped_log_probs, True 


def compute_sft_loss(model, prompt_input_ids, completion_input_ids, prompt_attention_mask, completion_attention_mask, model_device=None):
    """计算SFT损失，使用标准的模型forward pass和labels"""
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
    
    # 创建labels：prompt部分设为-100（忽略），completion部分为实际token
    prompt_len = prompt_input_ids.shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # 忽略prompt部分的loss
    
    # 前向传播并计算loss
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False
    )
    return outputs.loss


def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta=0.1, target_device=None):
    """计算DPO损失, 处理不同device上的tensor"""
    
    # 确定目标device (通常是policy model的device)
    if target_device is None:
        target_device = policy_chosen_logps.device
    
    # 将所有tensor移动到目标device
    policy_chosen_logps = policy_chosen_logps.to(target_device)
    policy_rejected_logps = policy_rejected_logps.to(target_device)
    ref_chosen_logps = ref_chosen_logps.to(target_device)
    ref_rejected_logps = ref_rejected_logps.to(target_device)
    
    # 计算log比率
    chosen_logratios = policy_chosen_logps - ref_chosen_logps  # bigger is better
    rejected_logratios = policy_rejected_logps - ref_rejected_logps  # smaller is better
    
    # DPO损失: -log(sigmoid(beta * (chosen_logratios - rejected_logratios)))
    logits = chosen_logratios - rejected_logratios
    losses = -F.logsigmoid(beta * logits)
    
    # 计算奖励（用于监控）
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    
    return losses.mean(), chosen_rewards, rejected_rewards


def grouped_dpo_loss(policy_chosen_grouped_logps, policy_rejected_grouped_logps,
                     ref_chosen_grouped_logps, ref_rejected_grouped_logps, 
                     beta=0.1, target_device=None):
    """计算分组DPO损失，返回每组各自的损失和奖励
    
    Args:
        policy_chosen_grouped_logps: policy model对chosen completion的分组log概率 [batch_size, num_groups]
        policy_rejected_grouped_logps: policy model对rejected completion的分组log概率 [batch_size, num_groups]
        ref_chosen_grouped_logps: reference model对chosen completion的分组log概率 [batch_size, num_groups]
        ref_rejected_grouped_logps: reference model对rejected completion的分组log概率 [batch_size, num_groups]
        beta: DPO温度参数
        target_device: 目标device
    
    Returns:
        tuple: (group_losses, chosen_group_rewards, rejected_group_rewards)
               group_losses: 每组的DPO损失 [batch_size, num_groups]
               chosen_group_rewards: 每组chosen的奖励 [batch_size, num_groups]
               rejected_group_rewards: 每组rejected的奖励 [batch_size, num_groups]
    """
    
    # 确定目标device (通常是policy model的device)
    if target_device is None:
        target_device = policy_chosen_grouped_logps.device
    
    # 将所有tensor移动到目标device
    policy_chosen_grouped_logps = policy_chosen_grouped_logps.to(target_device)
    policy_rejected_grouped_logps = policy_rejected_grouped_logps.to(target_device)
    ref_chosen_grouped_logps = ref_chosen_grouped_logps.to(target_device)
    ref_rejected_grouped_logps = ref_rejected_grouped_logps.to(target_device)
    
    # 计算每组的log比率
    chosen_group_logratios = policy_chosen_grouped_logps - ref_chosen_grouped_logps  # [batch_size, num_groups]
    rejected_group_logratios = policy_rejected_grouped_logps - ref_rejected_grouped_logps  # [batch_size, num_groups]
    
    # 计算每组的DPO损失: -log(sigmoid(beta * (chosen_logratios - rejected_logratios)))
    group_logits = chosen_group_logratios - rejected_group_logratios  # [batch_size, num_groups]
    group_losses = -F.logsigmoid(beta * group_logits)  # [batch_size, num_groups]
    
    # 计算每组的奖励（用于监控）
    chosen_group_rewards = beta * chosen_group_logratios.detach()  # [batch_size, num_groups]
    rejected_group_rewards = beta * rejected_group_logratios.detach()  # [batch_size, num_groups]
    
    return group_losses, chosen_group_rewards, rejected_group_rewards


def move_batch_to_device(batch, device):
    """将batch中的所有tensor移动到指定device"""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def create_lr_scheduler(optimizer, cfg):
    """创建学习率调度器
    
    Args:
        optimizer: 优化器
        cfg: 配置对象，应包含以下可选参数：
            - lr_scheduler_type: 调度器类型 ("cosine", "linear", "exponential", "step", "plateau", "none")
            - lr_warmup_steps: 预热步数 (仅用于linear和cosine)
            - lr_cosine_min_ratio: cosine调度器的最小学习率比例 (默认0.1)
            - lr_exponential_gamma: exponential调度器的衰减率 (默认0.95)
            - lr_step_size: step调度器的步长 (默认100)
            - lr_step_gamma: step调度器的衰减率 (默认0.5)
            - lr_plateau_patience: plateau调度器的耐心值 (默认10)
            - lr_plateau_factor: plateau调度器的衰减因子 (默认0.5)
    
    Returns:
        学习率调度器对象，如果不使用调度器则返回None
    """
    lr_scheduler_type = getattr(cfg, 'lr_scheduler_type', 'none')
    
    if lr_scheduler_type == 'none':
        return None
    
    elif lr_scheduler_type == 'cosine':
        # 余弦退火调度器
        max_steps = cfg.max_steps
        warmup_steps = getattr(cfg, 'lr_warmup_steps', 0)
        min_lr_ratio = getattr(cfg, 'lr_cosine_min_ratio', 0.1)
        
        if warmup_steps > 0:
            # 如果有预热，先用线性预热，再用余弦退火
            from torch.optim.lr_scheduler import SequentialLR
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps-warmup_steps, eta_min=cfg.learning_rate * min_lr_ratio)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=cfg.learning_rate * min_lr_ratio)
        
    elif lr_scheduler_type == 'linear':
        # 线性衰减调度器
        warmup_steps = getattr(cfg, 'lr_warmup_steps', 0)
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import SequentialLR
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_steps-warmup_steps)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])
        else:
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_steps)
    
    elif lr_scheduler_type == 'exponential':
        # 指数衰减调度器
        gamma = getattr(cfg, 'lr_exponential_gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    elif lr_scheduler_type == 'step':
        # 阶梯衰减调度器
        step_size = getattr(cfg, 'lr_step_size', 100)
        gamma = getattr(cfg, 'lr_step_gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif lr_scheduler_type == 'plateau':
        # 平台衰减调度器（基于loss）
        patience = getattr(cfg, 'lr_plateau_patience', 10)
        factor = getattr(cfg, 'lr_plateau_factor', 0.5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    
    else:
        raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")
    
    return scheduler


def train_dpo(model, ref_model, train_dataloader, cfg, if_not_demo=False):
    """DPO训练主循环，支持不同device上的模型"""

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+task{cfg.task_num}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    
    exp_id += f"--{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    # run_dir = os.path.join(cfg.run_root_dir, exp_id)
    adapter_dir = os.path.join(cfg.adapter_tmp_dir, exp_id)
    
    # Create directories
    # os.makedirs(run_dir, exist_ok=True)
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
    
    # 创建学习率调度器
    lr_scheduler = create_lr_scheduler(optimizer, cfg)
    if lr_scheduler is not None:
        print(f"Using learning rate scheduler: {getattr(cfg, 'lr_scheduler_type', 'none')}")
    else:
        print("No learning rate scheduler enabled")
    
    # 确保参考模型不更新
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_dpo_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_sft_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rewards_margin = deque(maxlen=cfg.grad_accumulation_steps)
    recent_chosen_logps = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rejected_logps = deque(maxlen=cfg.grad_accumulation_steps)
    recent_chosen_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rejected_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    
    # 训练循环
    with tqdm(total=cfg.max_steps, leave=False) as progress:
        model.train()
        # 创建无限迭代器，自动循环数据集
        def infinite_dataloader(dataloader):
            while True:
                for batch in dataloader:
                    yield batch
        
        infinite_loader = infinite_dataloader(train_dataloader)
        print(f"************Begin to train************")
        for batch_idx, batch in enumerate(infinite_loader):
            print(f"-----This is the {batch_idx}th batch----")
            # Check if we've reached max steps
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            if gradient_step_idx >= cfg.max_steps:
                break
                
            # 将batch移动到policy model的device
            policy_batch = move_batch_to_device(batch, policy_device)
            
            # 1. 计算策略模型的log概率 (在policy_device上)
            policy_chosen_logps, policy_chosen_grouped_logps, chosen_valid = compute_log_probs(
                model, 
                policy_batch['prompt_input_ids'], 
                policy_batch['chosen_input_ids'], 
                policy_batch['prompt_attention_mask'], 
                policy_batch['chosen_attention_mask'],
                is_chosen=True, is_policy=True, model_device=policy_device
            )
            
            if not chosen_valid:
                print(f"Skipping batch {batch_idx} due to invalid chosen completion")
                continue
                
            policy_rejected_logps, policy_rejected_grouped_logps, rejected_valid = compute_log_probs(
                model, 
                policy_batch['prompt_input_ids'], 
                policy_batch['rejected_input_ids'],
                policy_batch['prompt_attention_mask'], 
                policy_batch['rejected_attention_mask'],
                is_chosen=False, is_policy=True, model_device=policy_device
            )
            
            # 先计算SFT损失 (最大化chosen completion的概率)
            sft_loss_value = compute_sft_loss(
                model,
                policy_batch['prompt_input_ids'], 
                policy_batch['chosen_input_ids'], 
                policy_batch['prompt_attention_mask'], 
                policy_batch['chosen_attention_mask'],
                model_device=policy_device
            )
            
            # 如果rejected_valid == False，只训练SFT loss, 或跳过样本.
            if not rejected_valid:
                # print(f"Skipping batch {batch_idx} due to invalid rejected completion")
                # continue

                # 使用配置中的权重，如果没有则使用默认值
                sft_weight = getattr(cfg, 'sft_weight', 0.1)
                total_loss = sft_weight * sft_loss_value
                print(f"Batch {batch_idx}: rejected_valid=False, using only SFT loss: {sft_loss_value:.4f}")
                
                # 反向传播和优化
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                progress.update()
                
                # 记录指标（使用默认值填充缺失的DPO相关指标）
                recent_losses.append(total_loss.item())
                recent_dpo_losses.append(0.0)  # 没有DPO loss
                recent_sft_losses.append(sft_loss_value.item())
                recent_accuracies.append(0.0)  # 没有accuracy
                recent_rewards_margin.append(0.0)  # 没有reward margin
                recent_chosen_logps.append(policy_chosen_logps.mean().item() if policy_chosen_logps is not None else 0.0)
                recent_rejected_logps.append(0.0)  # 没有valid rejected logps
                recent_chosen_rewards.append(0.0)  # 没有rewards
                recent_rejected_rewards.append(0.0)
                
                # 记录到wandb
                if gradient_step_idx % 1 == 0 and if_not_demo:
                    wandb.log({
                        "total_loss": total_loss.item(),
                        "dpo_loss": 0.0,
                        "sft_loss": sft_loss_value.item(),
                        "accuracy": 0.0,
                        "reward_margin": 0.0,
                        "chosen_logps": policy_chosen_logps.mean().item() if policy_chosen_logps is not None else 0.0,
                        "rejected_logps": 0.0,
                        "chosen_rewards": 0.0,
                        "rejected_rewards": 0.0,
                        "distance": torch.mean(batch['distance']).item(),
                        "dpo_weight": 0.0,
                        "sft_weight": sft_weight,
                        "learning_rate": current_lr,
                        "start_idx": batch['start_idx']
                    }, step=gradient_step_idx)
                
                continue  # 跳过DPO计算部分
            
            # 2. 计算参考模型的log概率 (在ref_device上)
            with torch.no_grad():
                # 将batch移动到ref model的device
                ref_batch = move_batch_to_device(batch, ref_device)
                ref_chosen_logps, ref_chosen_grouped_logps, ref_chosen_valid = compute_log_probs(
                    ref_model, 
                    ref_batch['prompt_input_ids'], 
                    ref_batch['chosen_input_ids'],
                    ref_batch['prompt_attention_mask'], 
                    ref_batch['chosen_attention_mask'],
                    is_chosen=True, is_policy=False, model_device=ref_device
                )

                ref_rejected_logps, ref_rejected_grouped_logps, ref_rejected_valid = compute_log_probs(
                    ref_model, 
                    ref_batch['prompt_input_ids'], 
                    ref_batch['rejected_input_ids'],
                    ref_batch['prompt_attention_mask'], 
                    ref_batch['rejected_attention_mask'],
                    is_chosen=False, is_policy=False, model_device=ref_device
                ) 
            
            # 3. 计算DPO损失 (在policy_device上)
            dpo_loss_value, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=cfg.dpo_beta, target_device=policy_device
            )
            
            dpo_group_loss_value, chosen_group_rewards, rejected_group_rewards = grouped_dpo_loss(
                policy_chosen_grouped_logps, policy_rejected_grouped_logps,
                ref_chosen_grouped_logps, ref_rejected_grouped_logps,
                beta=cfg.dpo_beta, target_device=policy_device
            )

            # 计算分组准确率：对每组比较chosen_group_rewards和rejected_group_rewards
            group_accuracy = (chosen_group_rewards > rejected_group_rewards).float()[0]  # [batch_size, num_groups]
            print(f"Group_accuracy: {group_accuracy}")
            
            # 4. 根据group_accuracy调整损失计算策略
            # 使用配置中的权重，如果没有则使用默认值
            dpo_weight = getattr(cfg, 'dpo_weight', 1.0)
            sft_weight = getattr(cfg, 'sft_weight', 0.1)  # 默认SFT权重较小
            
            # 检查损失计算策略 (rejected_valid == True的情况)
            total_loss = torch.tensor(0.0, device=policy_device, requires_grad=True)
            
            # 判断group_accuracy
            if torch.all(group_accuracy == 1.0):
                # 全为1，只使用SFT loss
                total_loss = total_loss + sft_weight * sft_loss_value
                print(f"Batch {batch_idx}: All groups correct (accuracy=1), using only SFT loss: {sft_loss_value:.4f}")
            else:
                # 找到第一个0的位置，使用DPO loss
                zero_positions = torch.where(group_accuracy == 0.0)[0]
                if len(zero_positions) > 0:
                    first_zero_pos = int(zero_positions[0].item())
                    print(f"Batch {batch_idx}: First incorrect group at position {first_zero_pos}")
                    
                    # 只对第一个0的位置计算DPO loss + SFT loss
                    # 获取该组的DPO loss
                    first_zero_dpo_loss = dpo_group_loss_value[0, first_zero_pos]
                    
                    batch_total_loss = dpo_weight * first_zero_dpo_loss + sft_weight * sft_loss_value
                    total_loss = total_loss + batch_total_loss
                    print(f"Batch {batch_idx}: Using DPO loss from group {first_zero_pos}: {first_zero_dpo_loss:.4f} + SFT loss: {sft_loss_value:.4f}")
                else:
                    # 没有0，但也不是全1的情况（理论上不应该发生，因为group_accuracy只有0和1）
                    total_loss = total_loss + sft_weight * sft_loss_value
                    print(f"Batch {batch_idx}: No zero found but not all 1s, using only SFT loss: {sft_loss_value:.4f}")
            
            # print(f"batch_idx: {batch_idx}, total_loss: {total_loss:.4f}, dpo_loss: {dpo_loss_value:.4f}, sft_loss: {sft_loss_value:.4f}, distance: {batch['distance'].mean().item()}")
            # 6. 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新学习率调度器
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau需要传入loss值
                    lr_scheduler.step(total_loss.item())
                else:
                    # 其他调度器在每个gradient step后更新
                    if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                        lr_scheduler.step()
            
            progress.update()

            # 7. 计算指标
            accuracy = group_accuracy.float().mean()
            reward_margin = (chosen_group_rewards - rejected_group_rewards).mean()

            recent_losses.append(total_loss.item())
            
            # 根据是否使用了DPO loss来记录相应指标
            if torch.all(group_accuracy == 1.0):
                # 只使用了SFT loss的情况
                recent_dpo_losses.append(0.0)
                first_zero_dpo_loss = torch.tensor(0.0)  # 用于后续计算
            else:
                # 使用了DPO loss的情况
                zero_positions = torch.where(group_accuracy == 0.0)[0]
                if len(zero_positions) > 0:
                    first_zero_pos = int(zero_positions[0].item())
                    first_zero_dpo_loss = dpo_group_loss_value[0, first_zero_pos]
                    recent_dpo_losses.append(first_zero_dpo_loss.item())
                else:
                    recent_dpo_losses.append(0.0)
                    first_zero_dpo_loss = torch.tensor(0.0)
            
            recent_sft_losses.append(sft_loss_value.item())
            recent_accuracies.append(accuracy.item())
            recent_rewards_margin.append(reward_margin.item())
            
            # 确保logps不为None（应该在前面的检查中保证）
            assert policy_chosen_logps is not None, "policy_chosen_logps should not be None at this point"
            assert policy_rejected_logps is not None, "policy_rejected_logps should not be None at this point"
            
            recent_chosen_logps.append(policy_chosen_logps.mean().item())
            recent_rejected_logps.append(policy_rejected_logps.mean().item())
            recent_chosen_rewards.append(chosen_rewards.mean().item())
            recent_rejected_rewards.append(rejected_rewards.mean().item())

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_dpo_loss = sum(recent_dpo_losses) / len(recent_dpo_losses)
            smoothened_sft_loss = sum(recent_sft_losses) / len(recent_sft_losses)
            smoothened_accuracy = sum(recent_accuracies) / len(recent_accuracies)
            smoothened_rewards_margin = sum(recent_rewards_margin) / len(recent_rewards_margin)
            smoothened_chosen_logps = sum(recent_chosen_logps) / len(recent_chosen_logps)
            smoothened_rejected_logps = sum(recent_rejected_logps) / len(recent_rejected_logps)
            smoothened_chosen_rewards = sum(recent_chosen_rewards) / len(recent_chosen_rewards)
            smoothened_rejected_rewards = sum(recent_rejected_rewards) / len(recent_rejected_rewards)

            if gradient_step_idx % 1 == 0:
                if if_not_demo:
                    # 获取当前学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    wandb.log({
                        "total_loss": smoothened_loss,
                        "dpo_loss": smoothened_dpo_loss,
                        "sft_loss": smoothened_sft_loss,
                        "accuracy": smoothened_accuracy,
                        "reward_margin": smoothened_rewards_margin,
                        "chosen_logps": smoothened_chosen_logps,
                        "rejected_logps": smoothened_rejected_logps,
                        "chosen_rewards": smoothened_chosen_rewards,
                        "rejected_rewards": smoothened_rejected_rewards,
                        "distance": torch.mean(batch['distance']).item(),
                        "dpo_weight": dpo_weight,
                        "sft_weight": sft_weight,
                        "learning_rate": current_lr,
                        "start_idx": batch['start_idx']
                    },
                    step=gradient_step_idx,
                    )

            if gradient_step_idx % 10 == 0:
                # 为每个检查点创建独特的目录
                checkpoint_dir = os.path.join(adapter_dir, f"ckpt-{gradient_step_idx}")
                model.save_pretrained(checkpoint_dir)
                print(f"Saved adapter to {checkpoint_dir}, batch_idx: {batch_idx}")
                
            # 定期清理缓存（如果使用GPU）
            if gradient_step_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
    
    print(f"Training completed! Final adapter saved to: {adapter_dir}")
    return adapter_dir
