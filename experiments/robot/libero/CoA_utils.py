import os, torch
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import types
import draccus
import numpy as np
import tqdm
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

ACTION_DIM = 7

def spilt_chain_to_units(chain: torch.Tensor, unnorm_key):
    # Assert chain dimensions
    assert chain.shape[0] == 1, f"Expected batch size 1, got {chain.shape[0]}"
    unit_length = ACTION_DIM + 1  # Each unit has 7 action dims + 1 separator token
    # assert chain.shape[1] % unit_length == 0, f"Chain length {chain.shape[1]} is not divisible by unit length {unit_length}"
    
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

def process_action_unit(self, units:List[torch.Tensor], unnorm_key)->List[np.ndarray]:
    processed_units = []
    for unit in units:
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = unit[0,  :self.get_action_dim(unnorm_key)].cpu().numpy()
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
    self, input_ids: Optional[torch.LongTensor], unnorm_key: Optional[str], num_act_units: int = 100, **kwargs
    ) -> List[np.ndarray]:
    # print("Start Generating CoA")
    """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
    # Run VLA inference
    generated_ids = self.generate(input_ids, max_new_tokens=(self.get_action_dim(unnorm_key)+1)*num_act_units, **kwargs, do_sample=False)
    # assert (generated_ids.shape[1] - input_ids.shape[1]) % (ACTION_DIM + 1) == 0, f"Action shape {generated_ids.shape} is not divisible by {ACTION_DIM + 1}"
    # print("End Generating CoA")
    chain = generated_ids[:,input_ids.shape[1]:]
    # Check if EOS token (2) exists in the chain
    # assert chain.shape[1] % (ACTION_DIM + 1) == 0, f"Chain length {chain.shape} is not divisible by unit length {ACTION_DIM + 1}"
    units:List[torch.Tensor] = spilt_chain_to_units(chain, unnorm_key)
    processed_units:List[np.ndarray] = process_action_unit(self, units, unnorm_key)
    return processed_units


def test():
    print("tes11111t")