import os, re, pickle
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
import sys
import random
sys.path.append("../../")
from libero.libero import benchmark
from experiments.robot.openvla_utils import crop_and_resize
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_CoA,
)

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, cfg, winner_folder_path, task_suite_name, processor, device, model):
        self.winner_folder_path = winner_folder_path
        self.task_suite_name = task_suite_name
        self.data_path = os.path.join(self.winner_folder_path, self.task_suite_name)
        self.model = model
        self.processor = processor
        self.device = device
        self.cfg = cfg

        # Get list of all trajectory folders (only success trajectories)
        self.trajectory_folders = []
        for folder_name in os.listdir(self.data_path):
            match = re.search(r"task_(\d+)_episode_(\d+)_(failure|success)", folder_name)
            if match and match.group(3) == "success":
                self.trajectory_folders.append(folder_name)
        print(f"Found {len(self.trajectory_folders)} success trajectories")

    def get_winner_completion_ids(self, traj):
        action_chain = []
        action_sperate_token_id = 32001 # <A>
        for action in traj["action"]:
            action_chain.extend(action.tolist())  # Add the action tokens
            action_chain.append(action_sperate_token_id)  # Add separator token after each action
            # Ensure action_chain length is a multiple of 8
            assert len(action_chain) % 8 == 0, f"Action chain length {len(action_chain)} is not a multiple of 8"
        action_chain = torch.tensor(action_chain).to(self.device)
        winner_completion_ids = action_chain
        return winner_completion_ids

    def get_loser_completion_ids(self, traj, num_act_units = 10):
        ACTION_DIM = 7
        initial_state:dict = self.get_initial_state(traj, self.cfg.pretrained_checkpoint, self.processor, self.device)
        # loser_completion_ids = self.model.get_CoA(**initial_state, unnorm_key=self.cfg.unnorm_key, num_act_units=self.cfg.num_act_units)
        input_ids = initial_state['input_ids']

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        initial_state['input_ids'] = input_ids
        max_new_tokens=(self.model.get_action_dim(self.cfg.unnorm_key)+1)*num_act_units
        generated_ids = self.model.generate(max_new_tokens = max_new_tokens, **initial_state, do_sample = True, top_k = 1)
        assert (generated_ids.shape[1] - input_ids.shape[1]) % (ACTION_DIM + 1) == 0, f"Action shape {generated_ids.shape} is not divisible by {ACTION_DIM + 1}"
        chain = generated_ids[:,input_ids.shape[1]:]
        loser_completion_ids = chain
        return loser_completion_ids

    def prepare_trainer_input(self, idx):
        # trajectory = self.get_trajectory_data(idx)
        # winner_completion_ids = self.get_winner_completion_ids(trajectory)
        # loser_completion_ids = self.get_loser_completion_ids(trajectory)
        # return {"prompt_input_embeddings": winner_completion_ids, "chosen_input_ids": winner_completion_ids, "rejected_input_ids": loser_completion_ids}
        # 输入是embedding了, 之后要改的是concatenated_forward
        pass

    def get_trajectory_data(self, idx):
        folder_name = self.trajectory_folders[idx]
        
        # Parse folder name
        match = re.search(r"task_(\d+)_episode_(\d+)_(failure|success)", folder_name)
        if match:
            task_num = match.group(1)
            episode_num = match.group(2)
            result = match.group(3)
        else:
            raise ValueError(f"Invalid folder name: {folder_name}")

        task_description = get_task_description(self.task_suite_name, int(task_num))
        
        # Load trajectory data
        trajectory_folder_path = os.path.join(self.data_path, folder_name)
        trajectory = {
            "task_num": task_num, 
            "episode_num": episode_num, 
            "result": result,
            "task_description": task_description,
            "state": [], 
            "action": []
        }
        # Read all pickle files in the trajectory folder
        pkl_files = [f for f in os.listdir(trajectory_folder_path) if f.endswith(".pkl")]
        # Sort pkl files by step number
        pkl_files.sort(key=lambda x: int(re.search(r'step_(\d+)\.pkl', x).group(1)))
        start_idx = random.randint(0, len(pkl_files) - 5)

        for i in range(start_idx, len(pkl_files)):
            with open(os.path.join(trajectory_folder_path, pkl_files[i]), "rb") as f:
                data = pickle.load(f)
                state = data["obs"]
                action = data["action_ids"]
                trajectory["state"].append(state)
                trajectory["action"].append(action)
        return trajectory
    # trajectory is a dictionary containing:
    # - "task_num": str, task number extracted from folder name
    # - "episode_num": str, episode number extracted from folder name  
    # - "result": str, either "success" or "failure"
    # - "state": list, will contain observation dictionaries for each step
    # - "action": list, will contain action_ids (tokenized actions) for each step



    def get_initial_state(self, traj, base_vla_name, processor, device, resize_size = 224, center_crop=True)->dict:
        obs = traj["state"][0]
        task_label = traj["task_description"]
        img = get_libero_image(obs, resize_size)
        image = Image.fromarray(img)
        image = image.convert("RGB")
        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Crop and then resize back to original size
            image = crop_and_resize(image, crop_scale, batch_size)

            # Convert back to original data type
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            image = Image.fromarray(image.numpy())
            image = image.convert("RGB")
        # Build VLA prompt
        if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
            prompt = (
                f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
            )
        else:  # OpenVLA
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process inputs.
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        return inputs
    
    def __len__(self):
        return len(self.trajectory_folders)
    
    def __getitem__(self, idx) -> dict:
        trajectory = self.get_trajectory_data(idx)
        winner_completion_ids = self.get_winner_completion_ids(trajectory)
        loser_completion_ids = self.get_loser_completion_ids(trajectory)
        initial_state:dict = self.get_initial_state(trajectory, self.cfg.pretrained_checkpoint, self.processor, self.device)
        # loser_completion_ids = self.model.get_CoA(**initial_state, unnorm_key=self.cfg.unnorm_key, num_act_units=self.cfg.num_act_units)
        input_ids = initial_state['input_ids']
        return {"prompt_input_ids":input_ids, "chosen_input_ids": winner_completion_ids, "rejected_input_ids": loser_completion_ids}

        # folder_name = self.trajectory_folders[idx]
        
        # # Parse folder name
        # match = re.search(r"task_(\d+)_episode_(\d+)_(failure|success)", folder_name)
        # if match:
        #     task_num = match.group(1)
        #     episode_num = match.group(2)
        #     result = match.group(3)
        # else:
        #     raise ValueError(f"Invalid folder name: {folder_name}")

        # task_description = get_task_description(self.task_suite_name, int(task_num))
        
        # # Load trajectory data
        # trajectory_folder_path = os.path.join(self.data_path, folder_name)
        # trajectory = {
        #     "task_num": task_num, 
        #     "episode_num": episode_num, 
        #     "result": result,
        #     "task_description": task_description,
        #     "state": [], 
        #     "action": []
        # }

        # # Read all pickle files in the trajectory folder
        # pkl_files = [f for f in os.listdir(trajectory_folder_path) if f.endswith(".pkl")]
        # # Sort pkl files by step number
        # pkl_files.sort(key=lambda x: int(re.search(r'step_(\d+)\.pkl', x).group(1)))
        # start_idx = random.randint(0, len(pkl_files) - 5)
        # action_chain = []
        # action_sperate_token_id = 32001 # <A>
        # for i in range(start_idx, len(pkl_files)):
        #     with open(os.path.join(trajectory_folder_path, pkl_files[i]), "rb") as f:
        #         data = pickle.load(f)
        #         state = data["obs"]
        #         action = data["action_ids"]
        #         trajectory["state"].append(state)
        #         trajectory["action"].append(action)
        #         action_chain.extend(action.tolist())  # Add the action tokens
        #         action_chain.append(action_sperate_token_id)  # Add separator token after each action
        # # Ensure action_chain length is a multiple of 8
        # assert len(action_chain) % 8 == 0, f"Action chain length {len(action_chain)} is not a multiple of 8"
        # action_chain = torch.tensor(action_chain).to(self.device)
        # winner_completion_ids = action_chain

        # initial_state:dict = self.get_initial_state(trajectory, self.cfg.pretrained_checkpoint, self.processor, self.device)
        # # input_ids = torch.cat([initial_state['input_ids'][0], action_chain])

        # input_ids = initial_state['input_ids'][0]
        # pixel_values = initial_state['pixel_values']
        # attention_mask = torch.cat([initial_state['attention_mask'][0], torch.ones_like(action_chain)])
        # multimodal_embeddings, multimodal_attention_mask = prepare_multimodal_inputs(self.model, input_ids, attention_mask, pixel_values)
        # loser_completion_ids = get_loser_completion_ids(self.model, initial_state)

        # return {"input_embeddings": multimodal_embeddings,  "winner_completion_ids": winner_completion_ids, "loser_completion_ids": loser_completion_ids}
        
        # for file in pkl_files:
        #     with open(os.path.join(trajectory_folder_path, file), "rb") as f:
        #         data = pickle.load(f)
        #         state = data["obs"]
        #         action = data["action_ids"]
        #         trajectory["state"].append(state)
        #         trajectory["action"].append(action)

        # winner_trajectory = traj_preprocess(trajectory, 0, self.model, task_description, self.processor, self.model.device)
        # return trajectory       



def get_loser_completion_ids(model, initial_state):
    pass

def get_task_description(task_suite_name, task_id):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    return task.language

def get_initial_state(obs, base_vla_name, task_label, processor, device, resize_size = 224, center_crop=True)->dict:
    img = get_libero_image(obs, resize_size)
    image = Image.fromarray(img)
    image = image.convert("RGB")
    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")
    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    return inputs

def prepare_multimodal_inputs(
    model,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.LongTensor], torch.FloatTensor]:
    """
    处理多模态输入，将文本和图像融合为language model可以接受的格式
    
    Args:
        input_ids: 文本token ids [batch_size, seq_len]
        attention_mask: 注意力掩码 [batch_size, seq_len] 
        pixel_values: 图像像素值 [batch_size, channels, height, width]
        labels: 训练标签 [batch_size, seq_len]
    
    Returns:
        multimodal_embeddings: 融合后的多模态embeddings [batch_size, new_seq_len, hidden_dim]
        multimodal_attention_mask: 融合后的注意力掩码 [batch_size, new_seq_len]
        multimodal_labels: 融合后的标签 [batch_size, new_seq_len] 
        projected_patch_embeddings: 投影后的图像patch embeddings [batch_size, num_patches, hidden_dim]
    """
    IGNORE_INDEX = -100
    # 1. 视觉特征提取
    patch_features = model.vision_backbone(pixel_values)
    
    # 2. 将视觉特征投影到语言模型的隐藏维度
    projected_patch_embeddings = model.projector(patch_features)
    
    # 3. 为投影后的patch embeddings创建注意力掩码
    projected_patch_attention_mask = None
    if attention_mask is not None:
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
    
    # 4. 获取文本的input embeddings
    input_embeddings = model.get_input_embeddings()(input_ids)
    
    # 5. 构建多模态embeddings - 在<BOS> token (位置1)后插入图像patch embeddings
    # 格式: [<BOS>, <image_patches>, <remaining_text_tokens>]
    multimodal_embeddings = torch.cat(
        [input_embeddings[:1, :], projected_patch_embeddings[0], input_embeddings[1:, :]], dim=0
    )
    
    # 6. 构建多模态注意力掩码
    multimodal_attention_mask = None
    if attention_mask is not None:
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
        )
    
    # # 7. 构建多模态标签 - 为图像patch位置设置IGNORE_INDEX
    # multimodal_labels = None
    # if labels is not None:
    #     projected_patch_labels = torch.full(
    #         (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #         fill_value=IGNORE_INDEX,  # -100
    #         dtype=labels.dtype,
    #         device=labels.device,
    #     )
    #     multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
    
    return multimodal_embeddings, multimodal_attention_mask