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
    task_num: Optional[int] = None

    pretrained_checkpoint: Union[str, Path] = ""
    lora_path: str = ""
    base_vla_path: str = ""

    winner_trajectory_path: str = ""

    adapter_tmp_dir: str = ""
    run_root_dir: str = ""

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
    stream_length: int = 5
    
    # DPO loss weights
    dpo_weight: float = 1.0                          # Weight for DPO loss
    sft_weight: float = 0.1                          # Weight for SFT loss
    
    #################################################################################################################
    # Learning Rate Scheduler parameters
    #################################################################################################################
    lr_scheduler_type: str = "linear"                  # Options: "cosine", "linear", "exponential", "step", "plateau", "none"
    
    # Warmup parameters (for cosine and linear schedulers)
    lr_warmup_steps: int = 0                         # Number of warmup steps (0 = no warmup)
    
    # Cosine scheduler parameters
    lr_cosine_min_ratio: float = 0.1                 # Minimum LR ratio for cosine scheduler (final_lr = initial_lr * min_ratio)
    
    # Exponential scheduler parameters  
    lr_exponential_gamma: float = 0.95               # Decay rate for exponential scheduler
    
    # Step scheduler parameters
    lr_step_size: int = 100                          # Step size for step scheduler
    lr_step_gamma: float = 0.5                       # Decay factor for step scheduler
    
    # Plateau scheduler parameters
    lr_plateau_patience: int = 10                    # Patience for plateau scheduler
    lr_plateau_factor: float = 0.5                   # Decay factor for plateau scheduler
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
        # Set unnorm_key to task_suite_name if not explicitly set
        self.unnorm_key = self.task_suite_name
        
        # Set default paths if they are empty
        if not self.pretrained_checkpoint:
            self.pretrained_checkpoint = os.path.join(
                self.root_dir, 
                "openvla/FT_res/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
            )
        
        if not self.lora_path:
            self.lora_path = os.path.join(
                self.root_dir, 
                "openvla/adapter_tmp_dir/openvla-7b-finetuned-libero-10+libero_10_no_noops+b4+lr-0.0005+lora-r48+dropout-0.0--image_aug--2025-07-18_19-26-25"
            )
        
        if not self.base_vla_path:
            self.base_vla_path = os.path.join(self.root_dir, "HF_CACHE/openvla-7b-finetuned-libero-10")
            
        if not self.winner_trajectory_path:
            self.winner_trajectory_path = os.path.join(self.root_dir, "openvla/vla-scripts/DPO/winner_trajectory")
            
        if not self.adapter_tmp_dir:
            self.adapter_tmp_dir = os.path.join(self.root_dir, "openvla/DPO_adapter_tmp_dir")
            
        # if not self.run_root_dir:
        #     self.run_root_dir = os.path.join(self.root_dir, "openvla/DPO_res")
