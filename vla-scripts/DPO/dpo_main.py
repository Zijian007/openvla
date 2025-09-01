#!/usr/bin/env python3
"""
DPO Training Main Script

This script orchestrates the Direct Preference Optimization (DPO) training process
for Vision-Language-Action (VLA) models, specifically targeting OpenVLA models.

The script handles:
1. Model configuration and setup (policy and reference models)
2. Dataset creation and data loading  
3. DPO training loop with proper device management
4. Logging and checkpointing

Usage:
    python dpo_main.py [--config-overrides]
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directories to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Core imports
import torch
from torch.utils.data import DataLoader
from trl.trainer.dpo_trainer import DataCollatorForPreference
import numpy as np
from tqdm import tqdm
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_CoA,
)

# Local imports
from src.config import GenerateConfig
from src.model_utils import setup_vla_model_with_lora, setup_model_and_config, setup_logging_and_environment
from src.training_utils_prog import train_dpo
from src.data_process import TrajectoryDataset

# External imports  
from experiments.robot.robot_utils import get_model


def setup_data_loader(cfg, processor, model, env, task_suite, resize_size, human_prompt_template = "What action should the robot take to {lang}?"):
    """Setup the training data loader."""
    print("[*] Setting up dataset and data loader...")
    
    # Create dataset instance
    dataset = TrajectoryDataset(
        cfg, 
        cfg.winner_trajectory_path, 
        cfg.task_suite_name, 
        processor, 
        env, 
        task_suite,
        device=cfg.device, 
        model=model, 
        img_size=resize_size,
        stream_length=cfg.stream_length,
        task_num=cfg.task_num,
        if_fixed_stream_length = True,
        human_prompt_template=human_prompt_template
    )
    
    # Create data collator
    data_collator = DataCollatorForPreference(pad_token_id=processor.tokenizer.pad_token_id)
    
    # Create data loader
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    print(f"Dataset created with {len(dataset)} trajectory pairs")
    return train_dataloader


def main():
    """Main function to run DPO training."""
    parser = argparse.ArgumentParser(description="DPO Training for OpenVLA")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for policy model")
    parser.add_argument("--ref-device", type=str, default="cuda:0", help="Device for reference model")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--stream-length", type=int, default=5, help="Stream length for trajectory processing")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="openvla_CoA_DPO", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default="15652388600", help="Weights & Biases entity name")
    parser.add_argument("--run-id-note", type=str, default=None, help="Additional note for run ID")
    
    # Path configuration arguments (all optional with defaults)
    parser.add_argument("--root-dir", type=str, default="/mnt/sda/home/zijianwang", help="Root directory")
    parser.add_argument("--pretrained-checkpoint", type=str, default="", help="Path to pretrained checkpoint (uses default if empty)")
    parser.add_argument("--lora-path", type=str, default="", help="Path to LoRA adapter (uses default if empty)")
    parser.add_argument("--base-vla-path", type=str, default="", help="Path to base VLA model (uses default if empty)")
    parser.add_argument("--winner-trajectory-path", type=str, default="", help="Path to winner trajectory data (uses default if empty)")
    parser.add_argument("--task-num", type=int, default=None, help="Task number for training (uses default if empty)")
    parser.add_argument("--adapter-tmp-dir", type=str, default="", help="Directory for saving adapters (uses default if empty)")
    parser.add_argument("--run-root-dir", type=str, default="", help="Root directory for run outputs (uses default if empty)")
    
    args = parser.parse_args()

    print("[*] Starting OpenVLA DPO Training")
    
    # Check GPU availability and display information
    print(f"\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print("\n" + "="*50)
    print("Starting OpenVLA DPO Training")
    print("="*50)
    
    # Initialize configuration for policy model
    print("[*] Initializing configuration...")
    model_cfg = GenerateConfig(
        root_dir=args.root_dir,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dpo_beta=args.dpo_beta,
        stream_length=args.stream_length,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_id_note=args.run_id_note,
        grad_accumulation_steps=1,
        pretrained_checkpoint=args.pretrained_checkpoint,
        lora_path=args.lora_path,
        base_vla_path=args.base_vla_path,
        winner_trajectory_path=args.winner_trajectory_path,
        adapter_tmp_dir=args.adapter_tmp_dir,
        run_root_dir=args.run_root_dir,
        task_num=args.task_num
    )
    
    # Display configuration summary
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Policy Device: {model_cfg.device}")
    print(f"Reference Device: {args.ref_device}")
    print(f"Max Steps: {model_cfg.max_steps}")
    print(f"Batch Size: {model_cfg.batch_size}")
    print(f"Learning Rate: {model_cfg.learning_rate}")
    print(f"DPO Beta: {model_cfg.dpo_beta}")
    print(f"Stream Length: {model_cfg.stream_length}")
    print(f"Use WandB: {model_cfg.use_wandb}")
    print(f"Task Number: {model_cfg.task_num if model_cfg.task_num else 'All tasks'}")
    print("\nPath Configuration:")
    print(f"Root Dir: {model_cfg.root_dir}")
    print(f"Pretrained Checkpoint: {model_cfg.pretrained_checkpoint}")
    print(f"LoRA Path: {model_cfg.lora_path}")
    print(f"Winner Trajectory Path: {model_cfg.winner_trajectory_path}")
    print(f"Adapter Tmp Dir: {model_cfg.adapter_tmp_dir}")
    print(f"Run Root Dir: {model_cfg.run_root_dir}")
    print("="*50)
    
    # Setup policy model (with LoRA)
    print("[*] Loading policy model (with LoRA)...")
    print(f"Target device: {model_cfg.device}")
    model = setup_vla_model_with_lora(model_cfg)
    print(f"✓ Policy model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    # Setup logging and environment
    print("[*] Setting up logging and environment...")
    processor, log_file, task_suite, num_tasks_in_suite, resize_size = setup_logging_and_environment(model_cfg, model)
    
    # Setup task and environment
    print("[*] Setting up task and environment...")
    task = task_suite.get_task(model_cfg.task_num)
    env, task_description = get_libero_env(task, model_cfg.model_family, resolution=256)
    print(f"Task description: {task_description}")
    
    # Initialize configuration for reference model
    print("[*] Loading reference model...")
    print(f"Target device: {args.ref_device}")
    ref_config = GenerateConfig(
        root_dir=args.root_dir,
        device=args.ref_device,
        pretrained_checkpoint=args.pretrained_checkpoint,
        lora_path=args.lora_path,
        base_vla_path=args.base_vla_path,
        winner_trajectory_path=args.winner_trajectory_path,
        adapter_tmp_dir=args.adapter_tmp_dir,
        run_root_dir=args.run_root_dir
    )
    # ref_model = get_model(ref_config)
    ref_model = setup_vla_model_with_lora(ref_config)
    print(f"✓ Reference model loaded successfully")
    print(f"Model device: {next(ref_model.parameters()).device}")
    print(f"Model dtype: {next(ref_model.parameters()).dtype}")
    
    # Set reference model to eval mode and freeze parameters
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("✓ Reference model set to eval mode and frozen")
    
    # Setup data loader

    human_prompt_template = "What sequence of actions should the robot take to {lang}?"

    train_dataloader = setup_data_loader(model_cfg, processor, model, env, task_suite, resize_size, human_prompt_template)
    
    # # Verify setup with a test batch
    # print("[*] Verifying data loader setup...")
    # test_batch = next(iter(train_dataloader))
    # print(f"Batch keys: {test_batch.keys()}")
    # print(f"Chosen input shape: {test_batch['chosen_input_ids'].shape}")
    # print(f"Pixel values shape: {test_batch['pixel_values'].shape}")
    
    # Model loading summary
    print("\n" + "="*50)
    print("MODEL LOADING SUMMARY")
    print("="*50)
    print(f"Policy Model Device: {next(model.parameters()).device}")
    print(f"Reference Model Device: {next(ref_model.parameters()).device}")
    print(f"Policy Model Trainable: {sum(p.requires_grad for p in model.parameters())} params")
    print(f"Reference Model Trainable: {sum(p.requires_grad for p in ref_model.parameters())} params")
    print("Models loaded successfully!")
    print("="*50)
    
    # Optional: Clear cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")
    
    # Start DPO training
    print("[*] Starting DPO training...")
    print(f"Policy model device: {next(model.parameters()).device}")
    print(f"Reference model device: {next(ref_model.parameters()).device}")
    
    try:
        final_adapter_dir = train_dpo(
            model=model, 
            ref_model=ref_model, 
            train_dataloader=train_dataloader, 
            cfg=model_cfg, 
            if_not_demo=model_cfg.use_wandb
        )
        
        print(f"[*] Training completed successfully!")
        print(f"[*] Final adapter saved to: {final_adapter_dir}")
        
    except KeyboardInterrupt:
        print("\n[*] Training interrupted by user")
        
    except Exception as e:
        print(f"[*] Training failed with error: {e}")
        raise
        
    finally:
        # Clean up
        if log_file:
            log_file.close()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[*] Cleanup completed")


if __name__ == "__main__":
    main()
