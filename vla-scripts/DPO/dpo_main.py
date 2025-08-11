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

# Local imports
from src.config import GenerateConfig
from src.model_utils import setup_vla_model_with_lora, setup_model_and_config, setup_logging_and_environment
from src.training_utils import train_dpo
from src.data_process import TrajectoryDataset

# External imports  
from experiments.robot.robot_utils import get_model


def setup_data_loader(cfg, processor, model, resize_size):
    """Setup the training data loader."""
    print("[*] Setting up dataset and data loader...")
    
    # Create dataset instance
    dataset = TrajectoryDataset(
        cfg, 
        cfg.winner_trajectory_path, 
        cfg.task_suite_name, 
        processor, 
        device=cfg.device, 
        model=model, 
        img_size=resize_size,
        stream_length=5
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
    parser.add_argument("--device", type=str, default="cuda:2", help="Device for policy model")
    parser.add_argument("--ref-device", type=str, default="cuda:0", help="Device for reference model")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--run-id-note", type=str, default=None, help="Additional note for run ID")
    args = parser.parse_args()

    print("[*] Starting OpenVLA DPO Training")
    
    # Initialize configuration for policy model
    print("[*] Initializing configuration...")
    model_cfg = GenerateConfig(
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dpo_beta=args.dpo_beta,
        use_wandb=args.use_wandb,
        run_id_note=args.run_id_note,
        grad_accumulation_steps=1
    )
    
    # Setup policy model (with LoRA)
    print("[*] Loading policy model (with LoRA)...")
    model = setup_vla_model_with_lora(model_cfg)
    
    # Setup logging and environment
    print("[*] Setting up logging and environment...")
    processor, log_file, task_suite, num_tasks_in_suite, resize_size = setup_logging_and_environment(model_cfg, model)
    
    # Initialize configuration for reference model
    print("[*] Loading reference model...")
    ref_config = GenerateConfig(device=args.ref_device)
    ref_model = get_model(ref_config)
    
    # Setup data loader
    train_dataloader = setup_data_loader(model_cfg, processor, model, resize_size)
    
    # Verify setup with a test batch
    print("[*] Verifying data loader setup...")
    test_batch = next(iter(train_dataloader))
    print(f"Batch keys: {test_batch.keys()}")
    print(f"Chosen input shape: {test_batch['chosen_input_ids'].shape}")
    print(f"Pixel values shape: {test_batch['pixel_values'].shape}")
    
    # Test forward pass
    print("[*] Testing forward pass...")
    with torch.no_grad():
        pred_test = model.forward(
            input_ids=test_batch["prompt_input_ids"].to(model_cfg.device), 
            attention_mask=test_batch["chosen_attention_mask"].to(model_cfg.device), 
            pixel_values=test_batch["pixel_values"].to(model_cfg.device)
        )
    print(f"Model output keys: {pred_test.keys()}")
    
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
