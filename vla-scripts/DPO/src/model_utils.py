"""
Model setup and management utilities for DPO training.
"""

import os
import wandb
from typing import Tuple

from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    get_vla_via_lora,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from libero.libero import benchmark


def setup_model_and_config(cfg):
    """Setup and validate configuration, then load the model."""
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)
    
    return model


def setup_vla_model_with_lora(cfg):
    """Setup VLA model with LoRA configuration."""
    set_seed_everywhere(cfg.seed)
    model = get_vla_via_lora(cfg)
    return model


def setup_logging_and_environment(cfg, model) -> Tuple:
    """Setup logging and LIBERO environment.
    
    Returns:
        Tuple containing (processor, log_file, task_suite, num_tasks_in_suite, resize_size)
    """
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
