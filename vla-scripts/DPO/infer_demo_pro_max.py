#!/usr/bin/env python3
"""
Action Stream Inference Demo - Converted from Jupyter Notebook
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
import functools
import types
import time
from typing import Union, Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trl.trainer.dpo_trainer import DataCollatorForPreference
import numpy as np
import tqdm

warnings.filterwarnings('ignore')

# Add the parent directories to Python path for imports
current_dir = os.getcwd()
parent_dir = os.path.join(current_dir, "..", "..")
sys.path.append(parent_dir)
print(f"Added to Python path: {parent_dir}")

# Append current directory so that interpreter can find experiments.robot
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_CoA,
)
from experiments.robot.robot_utils import (
    get_CoA,
)

# Local imports
try:
    from infer_utils import predict_CoA, add_text_to_image
    from src.config import GenerateConfig
    from src.model_utils import setup_vla_model_with_lora, setup_logging_and_environment_base, create_log_file
    print("✓ Successfully imported local modules")
except ImportError as e:
    print(f"✗ Failed to import local modules: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)


def get_max_steps(task_suite_name: str) -> int:
    """Get maximum steps for different task suites."""
    max_steps_dict = {
        "libero_spatial": 220,  # longest training demo has 193 steps
        "libero_object": 280,   # longest training demo has 254 steps
        "libero_goal": 300,     # longest training demo has 270 steps
        "libero_10": 550,       # longest training demo has 505 steps
        "libero_90": 400,       # longest training demo has 373 steps
    }
    return max_steps_dict.get(task_suite_name, 300)


def create_config(args):
    """Create configuration object from command line arguments."""
    model_cfg = GenerateConfig(
        root_dir=args.root_dir,
        device=args.device,
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
    
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Policy Device: {model_cfg.device}")
    print(f"Stream Length: {model_cfg.stream_length}")
    print(f"Task Number: {model_cfg.task_num if model_cfg.task_num else 'All tasks'}")
    print("\nPath Configuration:")
    print(f"Root Dir: {model_cfg.root_dir}")
    print(f"Pretrained Checkpoint: {model_cfg.pretrained_checkpoint}")
    print(f"LoRA Path: {model_cfg.lora_path}")
    print(f"Winner Trajectory Path: {model_cfg.winner_trajectory_path}")
    print(f"Adapter Tmp Dir: {model_cfg.adapter_tmp_dir}")
    print(f"Run Root Dir: {model_cfg.run_root_dir}")
    print("="*50)
    
    return model_cfg


def load_model(model_cfg):
    """Load and setup the VLA model with LoRA."""
    print("Starting model loading...")
    print("This may take several minutes depending on model size and device speed.")
    print("\n" + "-"*30)
    
    # Load policy model with LoRA
    print("[1/2] Loading policy model (with LoRA)...")
    print(f"Target device: {model_cfg.device}")
    
    try:
        policy_model = setup_vla_model_with_lora(model_cfg)
        print(f"✓ Policy model loaded successfully")
        print(f"Model device: {next(policy_model.parameters()).device}")
        print(f"Model dtype: {next(policy_model.parameters()).dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy_model.parameters())
        trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
    except Exception as e:
        print(f"✗ Failed to load policy model: {e}")
        raise
    
    return policy_model


def setup_model_prediction(policy_model, top_k=1):
    """Setup CoA prediction method for the model."""
    # Create a function with default top_k
    predict_CoA_with_defaults = functools.partial(predict_CoA, top_k=top_k)
    
    # Add to model
    policy_model.get_CoA = types.MethodType(predict_CoA_with_defaults, policy_model)
    
    return top_k


def run_evaluation(model_cfg, policy_model, processor, task_suite, num_tasks_in_suite, resize_size, top_k):
    """Run the main evaluation loop."""
    human_prompt_template = "What sequence of actions should the robot take to {lang}?"
    
    # Record evaluation start time
    evaluation_start_time = time.time()
    
    # Create a main log file for overall evaluation
    main_log_file = create_log_file(model_cfg, "main_evaluation", None)
    
    def log_and_print(message, log_file=None):
        """Helper function to print and log simultaneously."""
        print(message)
        main_log_file.write(message + "\n")
        main_log_file.flush()
        if log_file and log_file != main_log_file:
            log_file.write(message + "\n")
            log_file.flush()
    
    log_and_print("[*] Starting evaluation...")
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_num_act_units = 20
    max_steps = get_max_steps(model_cfg.task_suite_name)
    
    # Dictionary to store results: {task_id: {num_act_units: (successes, episodes)}}
    results_matrix = {}
    
    # List to store all num_act_units values for final summary
    all_num_act_units = list(range(2, total_num_act_units, 2))
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if task_id < 3:
            continue
            
        # Get task
        task = task_suite.get_task(task_id)
        
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, model_cfg.model_family, resolution=256)

        # Initialize results storage for this task
        results_matrix[task_id] = {}
        
        # Start evaluation for different num_act_units
        task_episodes, task_successes = 0, 0
        
        log_and_print(f"\n{'='*80}")
        log_and_print(f"TASK {task_id}: {task_description}")
        log_and_print(f"{'='*80}")
        for num_act_units in tqdm.tqdm(range(2, total_num_act_units, 2), desc=f"Task {task_id} num_act_units"):  # Try from 2 to 20 units with step 2
            # if num_act_units != 10:
            #     continue
            
            log_and_print(f"\n{'-'*60}")
            log_and_print(f"Testing Task {task_id} with num_act_units: {num_act_units}")
            log_and_print(f"{'-'*60}")
            
            # Variables to track performance for this num_act_units
            num_act_episodes, num_act_successes = 0, 0
            
            # Create log file for this specific num_act_units
            log_file = create_log_file(model_cfg, num_act_units, task_id)
            log_file.write(f"Task {task_id}: {task_description}\n")
            log_file.write(f"num_act_units: {num_act_units}\n")
            log_file.write(f"{'-'*60}\n")
            log_file.flush()
            
            # Test multiple episodes for this num_act_units
            for episode_idx in range(model_cfg.num_trials_per_task):
                episode_start_time = time.time()
                log_and_print(f"Starting episode {episode_idx}...", log_file)
                
                # Reset environment and set initial state
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                
                total_length = 0
                replay_images = []
                CoA_step = 0
                done = False
                
                while total_length < max_steps + model_cfg.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if total_length < model_cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(model_cfg.model_family))
                            total_length += 1
                            continue

                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)  # np.array [224, 224, 3]
                        
                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }
                        
                        CoA: List[np.ndarray] = get_CoA(
                            model_cfg,
                            policy_model,
                            observation,
                            task_description,
                            processor=processor,
                            num_act_units=num_act_units,
                            human_prompt_template=human_prompt_template
                        )
                        
                        CoA_step += 1   # [array([0.002, -0.002, 0.000, -0.000, -0.000, -0.001, -1.000])]
                        current_CoA_length = len(CoA)
                        total_length = total_length + current_CoA_length
                        # Progress indicator (overwrite same line)
                        print(f"total_length:{total_length}/{max_steps}", end="\r")
                        
                        for unit_idx, action in enumerate(CoA):
                            obs, reward, done, info = env.step(action.tolist())
                            temp_img = get_libero_image(obs, resize_size)  # temp_img: np.ndarray, shape: (224, 224, 3)
                            temp_img = add_text_to_image(temp_img, num_act_units, CoA_step)
                            replay_images.append(temp_img)
                            
                        if done:
                            log_and_print(f"Success at task {task_id} -- episode {episode_idx} -- num_act_units:{num_act_units}, using t = {total_length}", log_file)
                            task_successes += 1
                            num_act_successes += 1
                            total_successes += 1
                            break
                            
                    except Exception as e:
                        log_and_print(f"Caught exception: {e}", log_file)
                        break
                
                # Calculate episode duration
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                
                # Update episode counters
                task_episodes += 1
                num_act_episodes += 1
                total_episodes += 1
                
                # Save rollout video for this episode
                save_rollout_video_CoA(
                    replay_images, episode_idx, success=done, task_description=task_description, 
                    log_file=log_file, num_act_units=num_act_units, task_id=task_id, 
                    task_suite_name=model_cfg.task_suite_name, top_k=top_k
                )
                
                # Calculate current success rate for this num_act_units
                current_success_rate = num_act_successes / num_act_episodes if num_act_episodes > 0 else 0
                
                # Log detailed episode results with timing and statistics
                log_and_print(f"Episode {episode_idx} completed:", log_file)
                log_and_print(f"  Success: {done}", log_file)
                log_and_print(f"  Duration: {episode_duration:.2f} seconds", log_file)
                log_and_print(f"  Current stats for num_act_units {num_act_units}:", log_file)
                log_and_print(f"    Episodes completed: {num_act_episodes}", log_file)
                log_and_print(f"    Successes: {num_act_successes}", log_file)
                log_and_print(f"    Success rate: {current_success_rate:.3f} ({num_act_successes}/{num_act_episodes})", log_file)
            
            # Store results for this combination
            results_matrix[task_id][num_act_units] = (num_act_successes, num_act_episodes)
            
            # Log results for this num_act_units
            success_rate_this_num_act = num_act_successes / num_act_episodes if num_act_episodes > 0 else 0
            log_and_print(f"\n*** RESULTS for Task {task_id}, num_act_units {num_act_units} ***", log_file)
            log_and_print(f"Success Rate: {success_rate_this_num_act:.3f} ({num_act_successes}/{num_act_episodes})", log_file)
            log_and_print(f"Global Progress: {total_episodes} episodes, {total_successes} successes ({total_successes / total_episodes * 100:.1f}%)", log_file)
            
            # Close log file for this num_act_units
            log_file.close()

        # Log final results for this task
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        log_and_print(f"\n*** TASK {task_id} SUMMARY ***")
        log_and_print(f"Task {task_id} overall success rate: {task_success_rate:.3f} ({task_successes}/{task_episodes})")
        log_and_print(f"Overall success rate: {float(total_successes) / float(total_episodes):.3f}")
        
        # Print task-specific summary
        log_and_print(f"\nTask {task_id} Results by num_act_units:")
        for num_act_units in all_num_act_units:
            if num_act_units in results_matrix[task_id]:
                successes, episodes = results_matrix[task_id][num_act_units]
                rate = successes / episodes if episodes > 0 else 0
                log_and_print(f"  num_act_units {num_act_units:2d}: {rate:.3f} ({successes}/{episodes})")
    
    # Calculate total evaluation time
    evaluation_end_time = time.time()
    total_evaluation_time = evaluation_end_time - evaluation_start_time
    
    # Print final comprehensive summary
    log_and_print(f"\n{'='*100}")
    log_and_print(f"FINAL EVALUATION SUMMARY")
    log_and_print(f"{'='*100}")
    log_and_print(f"Total Episodes: {total_episodes}")
    log_and_print(f"Total Successes: {total_successes}")
    log_and_print(f"Overall Success Rate: {total_successes / total_episodes:.3f}")
    log_and_print(f"Total Evaluation Time: {total_evaluation_time:.2f} seconds ({total_evaluation_time/60:.1f} minutes)")
    log_and_print(f"Average Time per Episode: {total_evaluation_time/total_episodes:.2f} seconds")
    log_and_print(f"\nDetailed Results Matrix:")
    
    # Build header line
    header_line = f"{'Task ID':<8}"
    for num_act_units in all_num_act_units:
        header_line += f"{'Units ' + str(num_act_units):<12}"
    log_and_print(header_line)
    log_and_print("-" * (8 + 12 * len(all_num_act_units)))
    
    for task_id in sorted(results_matrix.keys()):
        # Build each row as a string
        row_line = f"{task_id:<8}"
        for num_act_units in all_num_act_units:
            if num_act_units in results_matrix[task_id]:
                successes, episodes = results_matrix[task_id][num_act_units]
                rate = successes / episodes if episodes > 0 else 0
                row_line += f"{rate:.3f}({successes}/{episodes:<2})  "
            else:
                row_line += f"{'N/A':<12}"
        log_and_print(row_line)
    
    # Calculate and display average success rates by num_act_units
    log_and_print(f"\nAverage Success Rate by num_act_units:")
    log_and_print("-" * 50)
    for num_act_units in all_num_act_units:
        total_successes_for_units = 0
        total_episodes_for_units = 0
        for task_id in results_matrix:
            if num_act_units in results_matrix[task_id]:
                successes, episodes = results_matrix[task_id][num_act_units]
                total_successes_for_units += successes
                total_episodes_for_units += episodes
        
        if total_episodes_for_units > 0:
            avg_rate = total_successes_for_units / total_episodes_for_units
            log_and_print(f"num_act_units {num_act_units:2d}: {avg_rate:.3f} ({total_successes_for_units}/{total_episodes_for_units})")
        else:
            log_and_print(f"num_act_units {num_act_units:2d}: N/A")
    
    log_and_print(f"{'='*100}")
    
    # Close main log file
    main_log_file.close()


def main():
    """Main function to run the inference demo."""
    parser = argparse.ArgumentParser(description="OpenVLA CoA Inference Demo")
    
    # Training parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for policy model")
    parser.add_argument("--stream_length", type=int, default=10, help="Stream length for trajectory processing")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k for CoA prediction")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="openvla_CoA_DPO_demo", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="15652388600", help="WandB entity")
    parser.add_argument("--run_id_note", type=str, default="script_demo", help="Run ID note")
    
    # Path configuration
    parser.add_argument("--root_dir", type=str, default="/mnt/sda/home/zijianwang", help="Root directory")
    parser.add_argument("--pretrained_checkpoint", type=str, 
                       default="/mnt/sda/home/zijianwang/openvla/FT_res/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25",
                       help="Pretrained checkpoint path")
    parser.add_argument("--lora_path", type=str,
                       default="/mnt/sda/home/zijianwang/openvla/adapter_tmp_dir/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25",
                       help="LoRA path")
    parser.add_argument("--base_vla_path", type=str,
                       default="/mnt/sda/home/zijianwang/HF_CACHE/openvla-7b-finetuned-libero-10",
                       help="Base VLA path")
    parser.add_argument("--winner_trajectory_path", type=str,
                       default="/mnt/sda/home/zijianwang/openvla/vla-scripts/DPO/winner_trajectory",
                       help="Winner trajectory path")
    parser.add_argument("--adapter_tmp_dir", type=str,
                       default="/mnt/sda/home/zijianwang/openvla/DPO_adapter_tmp_dir",
                       help="Adapter temporary directory")
    parser.add_argument("--run_root_dir", type=str,
                       default="/mnt/sda/home/zijianwang/openvla/DPO_res",
                       help="Run root directory")
    
    # Task configuration
    parser.add_argument("--task_suite_name", type=str, default="libero_10", help="Task suite name")
    parser.add_argument("--task_num", type=int, default=1, help="Task number (set to None for all tasks)")
    
    args = parser.parse_args()
    
    # Check GPU availability
    print(f"\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    print("\n" + "="*50)
    print("Environment setup completed!")
    print("="*50)
    
    # Create configuration
    print("Creating configuration...")
    model_cfg = create_config(args)
    
    # Load model
    policy_model = load_model(model_cfg)
    
    # Setup CoA prediction
    top_k = setup_model_prediction(policy_model, args.top_k)
    
    # Setup logging and environment
    print("[*] Setting up logging and environment...")
    processor, task_suite, num_tasks_in_suite, resize_size = setup_logging_and_environment_base(model_cfg, policy_model)
    
    # Run evaluation
    run_evaluation(model_cfg, policy_model, processor, task_suite, num_tasks_in_suite, resize_size, top_k)


if __name__ == "__main__":
    main()
