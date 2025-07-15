export HF_HUB_CACHE=/mnt/sda/home/zijianwang/HF_CACHE
export CUDA_VISIBLE_DEVICES=3
# libero_spatial_no_noops, libero_object_no_noops, libero_goal_no_noops, libero_10_no_noops
python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_CoA.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/sda/home/zijianwang/openvla/modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir /mnt/sda/home/zijianwang/openvla/FT_res\
  --adapter_tmp_dir /mnt/sda/home/zijianwang/openvla/adapter_tmp_dir\
  --lora_rank 24 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla_CoA \
  --wandb_entity 15652388600 \
  --save_steps 1000 \
  --input_length 512