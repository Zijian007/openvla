export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_CoA.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /hdd/zijianwang/openvla/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --run_root_dir /hdd/zijianwang/openvla/FT_res\
  --adapter_tmp_dir /hdd/zijianwang/openvla/adapter_tmp_dir\
  --lora_rank 16 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla_CoA \
  --wandb_entity 15652388600 \
  --save_steps 1000