#!/bin/bash

# OpenVLA CoA Inference Demo Runner
# This script allows easy configuration of parameters for the inference demo

# =============================================================================
# CONFIGURATION PARAMETERS
# Modify these parameters as needed for your experiments
# =============================================================================

# ====== TRAINING PARAMETERS ======
DEVICE="cuda:1"                    # Device for policy model
STREAM_LENGTH=10                   # Stream length for trajectory processing
TOP_K=1                           # Top-k for CoA prediction

# ====== WANDB CONFIGURATION ======
USE_WANDB=false                   # Set to true to enable Weights & Biases logging
WANDB_PROJECT="openvla_CoA_DPO_demo"
WANDB_ENTITY="15652388600"
RUN_ID_NOTE="script_demo"

# ====== PATH CONFIGURATION ======
ROOT_DIR="/mnt/sda/home/zijianwang"

# Model paths
PRETRAINED_CHECKPOINT="${ROOT_DIR}/openvla/FT_res/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
LORA_PATH="${ROOT_DIR}/openvla/adapter_tmp_dir/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
BASE_VLA_PATH="${ROOT_DIR}/HF_CACHE/openvla-7b-finetuned-libero-10"

# Data paths
WINNER_TRAJECTORY_PATH="${ROOT_DIR}/openvla/vla-scripts/DPO/winner_trajectory"
ADAPTER_TMP_DIR="${ROOT_DIR}/openvla/DPO_adapter_tmp_dir"
RUN_ROOT_DIR="${ROOT_DIR}/openvla/DPO_res"

# ====== TASK CONFIGURATION ======
TASK_SUITE_NAME="libero_10"       # Task suite name                    # Task number (specific task to run)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Print configuration summary
echo "=================================================="
echo "OpenVLA CoA Inference Demo Configuration"
echo "=================================================="
echo "Device: $DEVICE"
echo "Stream Length: $STREAM_LENGTH"
echo "Top-K: $TOP_K"
echo "Task Suite: $TASK_SUITE_NAME"
echo "Use WandB: $USE_WANDB"
echo ""
echo "Paths:"
echo "  Root Dir: $ROOT_DIR"
echo "  Pretrained Checkpoint: $PRETRAINED_CHECKPOINT"
echo "  LoRA Path: $LORA_PATH"
echo "  Base VLA Path: $BASE_VLA_PATH"
echo "=================================================="
echo ""

# Change to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build command arguments
PYTHON_ARGS=(
    --device "$DEVICE"
    --stream_length "$STREAM_LENGTH"
    --top_k "$TOP_K"
    --wandb_project "$WANDB_PROJECT"
    --wandb_entity "$WANDB_ENTITY"
    --run_id_note "$RUN_ID_NOTE"
    --root_dir "$ROOT_DIR"
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
    --lora_path "$LORA_PATH"
    --base_vla_path "$BASE_VLA_PATH"
    --winner_trajectory_path "$WINNER_TRAJECTORY_PATH"
    --adapter_tmp_dir "$ADAPTER_TMP_DIR"
    --run_root_dir "$RUN_ROOT_DIR"
    --task_suite_name "$TASK_SUITE_NAME"
)

# Add wandb flag if enabled
if [ "$USE_WANDB" = true ]; then
    PYTHON_ARGS+=(--use_wandb)
fi

# Set environment variables for tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Check if Python script exists
if [ ! -f "infer_demo_pro_max.py" ]; then
    echo "Error: infer_demo_pro_max.py not found in current directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Run the Python script
echo "Starting inference demo..."
echo "Command: python infer_demo_pro_max.py ${PYTHON_ARGS[*]}"
echo ""

python infer_demo_pro_max.py "${PYTHON_ARGS[@]}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Inference demo completed successfully!"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "Inference demo failed with exit code $?"
    echo "=================================================="
    exit 1
fi
