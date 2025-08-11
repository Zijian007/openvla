"""
Configuration classes for DPO training.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class GenerateConfig:
    """Configuration class for DPO training and model setup."""

    # fmt: off
    vla_path: str = "openvla/openvla-7b" 
    root_dir: str = "/mnt/sda/home/zijianwang"

    #################################################################################################################
    # LoRA parameters
    #################################################################################################################
    use_lora: bool = True
    lora_rank: int = 48
    lora_dropout: float = 0.0
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family

    dataset_name: str = "libero_10_no_noops"

    pretrained_checkpoint: Union[str, Path] = None
    lora_path: str = None
    base_vla_path: str = None

    winner_trajectory_path: str = None

    adapter_tmp_dir: str = None
    run_root_dir: str = None

    #################################################################################################################
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    #################################################################################################################
    # Training parameters
    #################################################################################################################
    batch_size: int = 4
    grad_accumulation_steps: int = 1
    learning_rate: float = 0.0005
    max_steps: int = 10000
    dpo_beta: float = 0.1
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    unnorm_key = task_suite_name
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "openvla_CoA_DPO"        # Name of W&B project to log to
    wandb_entity: str = "15652388600"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    device: str = "cuda:2"

    # fmt: on

    def __post_init__(self):
        """Initialize derived paths after object creation."""
        if self.pretrained_checkpoint is None:
            self.pretrained_checkpoint = os.path.join(
                self.root_dir, 
                "openvla/FT_res/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
            )
        
        if self.lora_path is None:
            self.lora_path = os.path.join(
                self.root_dir, 
                "openvla/adapter_tmp_dir/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
            )
        
        if self.base_vla_path is None:
            self.base_vla_path = os.path.join(self.root_dir, "HF_CACHE/openvla-7b-finetuned-libero-10")
            
        if self.winner_trajectory_path is None:
            self.winner_trajectory_path = os.path.join(self.root_dir, "openvla/vla-scripts/DPO/winner_trajectory")
            
        if self.adapter_tmp_dir is None:
            self.adapter_tmp_dir = os.path.join(self.root_dir, "openvla/adapter_tmp_dir")
            
        if self.run_root_dir is None:
            self.run_root_dir = os.path.join(self.root_dir, "openvla/DPO_res")

        # Set unnorm_key to task_suite_name if not explicitly set
        self.unnorm_key = self.task_suite_name
