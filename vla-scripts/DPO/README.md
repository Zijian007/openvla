# DPO Training for OpenVLA

This directory contains the refactored DPO (Direct Preference Optimization) training code for OpenVLA models, organized from the original Jupyter notebook.

## Project Structure

```
DPO/
├── dpo_main.py              # Main training script
├── test_imports.py          # Import verification script  
├── README.md               # This file
├── src/
│   ├── config.py           # Configuration classes
│   ├── model_utils.py      # Model setup and management utilities
│   ├── training_utils.py   # DPO training functions and loss computation
│   └── data_process.py     # Dataset classes (existing)
└── winner_trajectory/      # Training data directory
```

## Quick Start

1. **Run DPO training** using the shell script (recommended):
   ```bash
   cd /mnt/sda/home/zijianwang/openvla
   bash scripts_dpo_10.sh
   ```

2. **Run manually** with default paths (minimal command):
   ```bash
   cd /mnt/sda/home/zijianwang/openvla
   python vla-scripts/DPO/dpo_main.py --use-wandb
   ```

3. **Run manually** with custom parameters (all paths will use defaults if not specified):
   ```bash
   cd /mnt/sda/home/zijianwang/openvla
   python vla-scripts/DPO/dpo_main.py \
     --device cuda:3 \
     --ref-device cuda:0 \
     --max-steps 10000 \
     --batch-size 8 \
     --stream-length 20 \
     --use-wandb \
     --wandb-project "openvla_CoA_DPO" \
     --wandb-entity "15652388600"
   ```

## Configuration

The main configuration is handled by the `GenerateConfig` class in `src/config.py`. Key parameters include:

- **Model paths**: Automatically set to defaults, can be overridden via command line arguments
- **LoRA settings**: `use_lora=True`, `lora_rank=48`, `lora_dropout=0.0`
- **Training**: `batch_size=4`, `learning_rate=0.0005`, `max_steps=10000`
- **DPO**: `dpo_beta=0.1` (controls the strength of preference optimization)

## Command Line Options

### Training Parameters
- `--device`: GPU device for policy model (default: cuda:2)
- `--ref-device`: GPU device for reference model (default: cuda:0)  
- `--max-steps`: Maximum training steps (default: 10000)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 0.0005)
- `--dpo-beta`: DPO beta parameter (default: 0.1)
- `--stream-length`: Stream length for trajectory processing (default: 5)
- `--use-wandb`: Enable Weights & Biases logging
- `--wandb-project`: Weights & Biases project name (default: openvla_CoA_DPO)
- `--wandb-entity`: Weights & Biases entity name (default: 15652388600)
- `--run-id-note`: Additional note for experiment tracking

### Path Configuration (All Optional with Smart Defaults)
- `--root-dir`: Root directory (default: /mnt/sda/home/zijianwang)
- `--pretrained-checkpoint`: Path to pretrained checkpoint (auto-generated from root-dir if empty)
- `--lora-path`: Path to LoRA adapter (auto-generated from root-dir if empty)
- `--base-vla-path`: Path to base VLA model (auto-generated from root-dir if empty)
- `--winner-trajectory-path`: Path to winner trajectory data (auto-generated from root-dir if empty)
- `--adapter-tmp-dir`: Directory for saving adapters (auto-generated from root-dir if empty)
- `--run-root-dir`: Root directory for run outputs (auto-generated from root-dir if empty)

## Key Features

- **Multi-GPU support**: Policy and reference models can be on different GPUs
- **LoRA training**: Efficient fine-tuning with Low-Rank Adaptation
- **Automatic checkpointing**: Models saved every 10 gradient steps
- **Comprehensive logging**: Support for both local logging and W&B
- **Gradient accumulation**: Configurable for larger effective batch sizes
- **Memory management**: Automatic GPU cache clearing

## Training Process

1. **Model Setup**: Loads policy model with LoRA and reference model
2. **Data Loading**: Creates trajectory dataset with preference pairs
3. **DPO Training**: Alternates between:
   - Computing log probabilities for chosen/rejected completions
   - Calculating DPO loss with policy vs reference model comparison
   - Backpropagation and gradient updates
   - Periodic checkpointing and logging

## Output

- **Adapters**: Saved to `{root_dir}/openvla/adapter_tmp_dir/{experiment_id}/`
- **Logs**: Local logs in `./experiments/logs/`
- **W&B**: Online tracking if `--use-wandb` is enabled

## Notes

- Ensure trajectory data exists in `winner_trajectory/libero_10/`
- The script automatically handles device placement for multi-GPU setups
- Training can be interrupted with Ctrl+C and will save the current state
- Memory usage is optimized with periodic cache clearing
