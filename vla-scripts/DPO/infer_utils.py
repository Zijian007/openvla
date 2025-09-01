import cv2
import torch
import numpy as np
from typing import Optional, List
import os, sys
sys.path.append("../..")
import wandb
from libero.libero import benchmark


from experiments.robot.openvla_utils import get_processor, get_input
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



from src.config import GenerateConfig


ACTION_DIM = 7
def add_text_to_image(temp_img, num_act_units, CoA_step):
    """Add text overlay to image showing length and step number.
    
    Args:
        temp_img (np.ndarray): Input image of shape (224, 224, 3)
        num_act_units (int): Number of action units
        CoA_step (int): Current step number
        
    Returns:
        np.ndarray: Image with text overlay
    """
    img = temp_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"length: {num_act_units}, step: {CoA_step}"
    
    # Get text size to position it in upper right
    (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 1)
    
    # Position text 10 pixels from right and top edges
    text_x = img.shape[1] - text_width - 10
    text_y = text_height + 10
    
    # Add white text with black outline for visibility
    cv2.putText(img, text, (text_x, text_y), font, 0.5, (0,0,0), 2)
    cv2.putText(img, text, (text_x, text_y), font, 0.5, (255,255,255), 1)
    
    return img

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
        # assert unit[:,-1] == 32001, f"Unit {i} does not end with separator token 32001"
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
    self, input_ids: Optional[torch.LongTensor], unnorm_key: Optional[str], num_act_units: int = 100, top_k: int = 2, **kwargs: str
) -> List[np.ndarray]:
    """Predict Chain of Actions (CoA) using the VLA model."""
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
    
    # Run VLA inference
    # print(f"input_ids: {input_ids}")

    generated_ids = self.generate(input_ids, max_new_tokens=(self.get_action_dim(unnorm_key)+1)*num_act_units, **kwargs, do_sample=True, top_k = top_k)
    # print(f"generated_ids: {generated_ids}")
    assert (generated_ids.shape[1] - input_ids.shape[1]) % (ACTION_DIM + 1) == 0, f"Action shape {generated_ids.shape} is not divisible by {ACTION_DIM + 1}"
    chain = generated_ids[:,input_ids.shape[1]:]
    assert chain.shape[1] % (ACTION_DIM + 1) == 0, f"Chain length {chain.shape} is not divisible by unit length {ACTION_DIM + 1}"
    
    units: List[torch.Tensor] = spilt_chain_to_units(chain, unnorm_key)
    processed_units: List[np.ndarray] = process_action_unit(self, units, unnorm_key)
    return processed_units

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