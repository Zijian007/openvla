export HF_HUB_CACHE=/mnt/sda/home/zijianwang/HF_CACHE
export CUDA_VISIBLE_DEVICES=3

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /mnt/sda/home/zijianwang/openvla/FT_res/openvla-7b+libero_goal_no_noops+b4+lr-0.0005+lora-r24+dropout-0.0--image_aug \
  --task_suite_name libero_goal \
  --center_crop True