export HF_HUB_CACHE=/mnt/sda/home/zijianwang/HF_CACHE
export CUDA_VISIBLE_DEVICES=3

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /mnt/sda/home/zijianwang/HF_CACHE/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --use_wandb True \
  --wandb_project openvla_eval_libero_10 \
  --wandb_entity 15652388600 \