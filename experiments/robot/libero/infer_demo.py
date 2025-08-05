#!/usr/bin/env python3
"""
OpenVLA LIBERO Inference Demo
Converted from Jupyter notebook to standalone Python script.
"""

import os
import sys
import torch
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    save_rollout_video_CoA
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_CoA,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    """Configuration for OpenVLA inference on LIBERO tasks."""
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/mnt/sda/home/zijianwang/openvla/FT_res/openvla-7b+libero_goal_no_noops+b4+lr-0.0005+lora-r24+dropout-0.0--image_aug--2025-07-08_15-39-30"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"             # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    device: str = "cuda:2"                             # Device to use for inference


# Constants
ACTION_DIM = 7


def spilt_chain_to_units(chain: torch.Tensor, unnorm_key):
    """Split action chain into individual action units."""
    # Assert chain dimensions
    assert chain.shape[0] == 1, f"Expected batch size 1, got {chain.shape[0]}"
    unit_length = ACTION_DIM + 1  # Each unit has 7 action dims + 1 separator token
    assert chain.shape[1] % unit_length == 0, f"Chain length {chain.shape[1]} is not divisible by unit length {unit_length}"
    
    # Split chain into units
    num_units = chain.shape[1] // unit_length
    units = []
    for i in range(num_units):
        start_idx = i * unit_length
        end_idx = start_idx + unit_length
        unit = chain[:, start_idx:end_idx]
        units.append(unit)
        assert unit[:,-1] == 32001, f"Unit {i} does not end with separator token 32001"
    return units


def process_action_unit(self, units: List[torch.Tensor], unnorm_key) -> List[np.ndarray]:
    """Process action units and convert to continuous actions."""
    processed_units = []
    for unit in units:
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = unit[0, :self.get_action_dim(unnorm_key)].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        processed_units.append(action)
    return processed_units


def predict_CoA(
    self, input_ids: Optional[torch.LongTensor], unnorm_key: Optional[str], num_act_units: int = 100, **kwargs: str
) -> List[np.ndarray]:
    """Predict Chain of Actions (CoA) using the VLA model."""
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
    
    # Run VLA inference
    generated_ids = self.generate(input_ids, max_new_tokens=(self.get_action_dim(unnorm_key)+1)*num_act_units, **kwargs)
    assert (generated_ids.shape[1] - input_ids.shape[1]) % (ACTION_DIM + 1) == 0, f"Action shape {generated_ids.shape} is not divisible by {ACTION_DIM + 1}"
    
    chain = generated_ids[:,input_ids.shape[1]:]
    assert chain.shape[1] % (ACTION_DIM + 1) == 0, f"Chain length {chain.shape} is not divisible by unit length {ACTION_DIM + 1}"
    
    units: List[torch.Tensor] = spilt_chain_to_units(chain, unnorm_key)
    processed_units: List[np.ndarray] = process_action_unit(self, units, unnorm_key)
    return processed_units


def execute_CoA(env, CoA):
    """Execute Chain of Actions in the environment."""
    for action in CoA:
        obs, reward, done, info = env.step(action.tolist())
        if done:
            return True
    return False


def get_max_steps(task_suite_name: str) -> int:
    """Get maximum steps for different task suites."""
    max_steps_dict = {
        "libero_spatial": 220,  # longest training demo has 193 steps
        "libero_object": 280,   # longest training demo has 254 steps
        "libero_goal": 300,     # longest training demo has 270 steps
        "libero_10": 520,       # longest training demo has 505 steps
        "libero_90": 400,       # longest training demo has 373 steps
    }
    return max_steps_dict.get(task_suite_name, 300)


def setup_model_and_config(cfg: GenerateConfig):
    """Setup and validate configuration, then load the model."""
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)
    
    # Add CoA prediction method to model
    model.get_CoA = types.MethodType(predict_CoA, model)
    
    return model


def setup_logging_and_environment(cfg: GenerateConfig, model):
    """Setup logging and LIBERO environment."""
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    return processor, log_file, task_suite, num_tasks_in_suite, resize_size


def run_evaluation(cfg: GenerateConfig, model, processor, log_file, task_suite, num_tasks_in_suite, resize_size):
    """Run the main evaluation loop."""
    # Start evaluation
    total_episodes, total_successes = 0, 0
    max_steps = get_max_steps(cfg.task_suite_name)

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if task_id != 1: 
            continue
            
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if episode_idx != 0:  # Currently only running first episode
                break
                
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            for num_act_units in range(35, 36):  # Try from 1 to 40 units
                print(f"num_act_units:{num_act_units}")
                total_length = 0
                replay_images = []
                
                # Reset environment
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                while total_length < max_steps + cfg.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if total_length < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            total_length += 1
                            continue

                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }
                        
                        CoA: List[np.ndarray] = get_CoA(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            num_act_units=num_act_units,
                        )
                        
                        current_CoA_length = len(CoA)
                        total_length = total_length + current_CoA_length
                        print(f"total_length:{total_length}/{max_steps}", end="\r")
                        
                        for unit_idx, action in enumerate(CoA):
                            obs, reward, done, info = env.step(action.tolist())
                            temp_img = get_libero_image(obs, resize_size)
                            replay_images.append(temp_img)
                            
                    except Exception as e:
                        print(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        break

                save_rollout_video_CoA(
                    replay_images, total_episodes, success=done, task_description=task_description, 
                    log_file=log_file, num_act_units=num_act_units, task_id=task_id, task_suite_name=cfg.task_suite_name
                )

    log_file.close()


def main():
    """Main function to run the OpenVLA LIBERO inference demo."""
    print("[*] Starting OpenVLA LIBERO Inference Demo")
    
    # Initialize configuration
    cfg = GenerateConfig()
    
    # Setup model and configuration
    print("[*] Loading model and setting up configuration...")
    model = setup_model_and_config(cfg)
    
    # Setup logging and environment
    print("[*] Setting up logging and environment...")
    processor, log_file, task_suite, num_tasks_in_suite, resize_size = setup_logging_and_environment(cfg, model)
    
    # Run evaluation
    print("[*] Starting evaluation...")
    run_evaluation(cfg, model, processor, log_file, task_suite, num_tasks_in_suite, resize_size)
    
    print("[*] Evaluation completed!")


if __name__ == "__main__":
    main() 