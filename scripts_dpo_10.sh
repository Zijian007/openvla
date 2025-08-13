export HF_HUB_CACHE=/mnt/sda/home/zijianwang/HF_CACHE

# Define path variables
ROOT_DIR="/mnt/sda/home/zijianwang"
PRETRAINED_CHECKPOINT="${ROOT_DIR}/openvla/FT_res/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
LORA_PATH="${ROOT_DIR}/openvla/adapter_tmp_dir/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
BASE_VLA_PATH="${ROOT_DIR}/HF_CACHE/openvla-7b-finetuned-libero-10"
WINNER_TRAJECTORY_PATH="${ROOT_DIR}/openvla/vla-scripts/DPO/winner_trajectory"
ADAPTER_TMP_DIR="${ROOT_DIR}/openvla/DPO_adapter_tmp_dir"
RUN_ROOT_DIR="${ROOT_DIR}/openvla/DPO_res"

# Define WandB configuration
WANDB_PROJECT="openvla_CoA_DPO"
WANDB_ENTITY="15652388600"

# libero_spatial_no_noops, libero_object_no_noops, libero_goal_no_noops: lora24, libero_10_no_noops
python vla-scripts/DPO/dpo_main.py \
  --device cuda:3 \
  --ref-device cuda:0 \
  --max-steps 10000 \
  --batch-size 2 \
  --stream-length 20 \
  --task-num 1 \
  --root-dir "${ROOT_DIR}" \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --lora-path "${LORA_PATH}" \
  --base-vla-path "${BASE_VLA_PATH}" \
  --winner-trajectory-path "${WINNER_TRAJECTORY_PATH}" \
  --adapter-tmp-dir "${ADAPTER_TMP_DIR}" \
  --run-root-dir "${RUN_ROOT_DIR}" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}"
