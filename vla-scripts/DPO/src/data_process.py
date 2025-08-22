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

from experiments.robot.robot_utils import (
    invert_gripper_action,
    normalize_gripper_action
)

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, cfg, winner_folder_path, task_suite_name, processor, env, task_suite, device, model, img_size = 224, stream_length = 10, task_num = None, if_offline = False):
        self.winner_folder_path = winner_folder_path
        self.task_suite_name = task_suite_name
        self.data_path = os.path.join(self.winner_folder_path, self.task_suite_name)
        self.model = model
        self.processor = processor
        self.device = device
        self.cfg = cfg
        self.stream_length = stream_length
        self.img_size = img_size
        self.task_num = task_num
        self.env = env
        self.task_suite = task_suite
        self.if_offline = if_offline

        if self.if_offline == True:
            self.complete_loser_trajectory = self.get_complete_loser_trajectory(self.data_path)

            
        # Get list of all trajectory folders (only success trajectories)
        self.trajectory_folders = []
        self.task_trajectories = {}  # Dictionary to store trajectories by task number
        for folder_name in os.listdir(self.data_path):
            match = re.search(r"task_(\d+)_episode_(\d+)_(failure|success)", folder_name)
            if match and match.group(3) == "success":
                task_num = match.group(1)
                if task_num not in self.task_trajectories:
                    self.task_trajectories[task_num] = []
                self.task_trajectories[task_num].append(folder_name)
                self.trajectory_folders.append(folder_name)
        print(f"Found {len(self.trajectory_folders)} success trajectories")
        print(f"Task distribution: {[(task, len(folders)) for task, folders in self.task_trajectories.items()]}")

    def get_winner_completion_ids(self, traj, start_idx):
        action_chain = []
        state_chain = []
        action_sperate_token_id = 32001 # <A>
        for action, state in zip(traj["action"][start_idx:start_idx + self.stream_length], traj["state"][start_idx:start_idx + self.stream_length]):
            # assert action[-1] == 32001, f"Action {action} does not end with separator token 32001 in winner trajectory sampling"
            action_chain.extend(action.tolist())  # Add the action tokens
            # action_chain.append(action_sperate_token_id)  # Add separator token after each action
            state_chain.append(state)
            # Ensure action_chain length is a multiple of 8
            assert len(action_chain) % 8 == 0, f"Action chain length {len(action_chain)} is not a multiple of 8"

        action_chain = torch.tensor(action_chain).to(self.device)
        winner_completion_ids = action_chain
        return winner_completion_ids, state_chain

    def get_loser_completion_ids(self, initial_state):
        ACTION_DIM = 7
        # loser_completion_ids = self.model.get_CoA(**initial_state, unnorm_key=self.cfg.unnorm_key, num_act_units=self.cfg.num_act_units)
        input_ids = initial_state['input_ids']

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        initial_state['input_ids'] = input_ids
        max_new_tokens = (self.model.get_action_dim(self.cfg.unnorm_key) + 1)*self.stream_length
        assert self.model.get_action_dim(self.cfg.unnorm_key) == 7, f"Action dim {self.model.get_action_dim(self.cfg.unnorm_key)} is not 7"
        generated_ids = self.model.generate(max_new_tokens = max_new_tokens, **initial_state, do_sample = True, top_k = 1)
        # assert (generated_ids.shape[1] - input_ids.shape[1]) % (ACTION_DIM + 1) == 0, f"Action shape {generated_ids.shape} is not divisible by {ACTION_DIM + 1}"
        chain = generated_ids[:,input_ids.shape[1]:]
        loser_completion_ids = chain

        # state_chain = execute_action_chain_ids(self.cfg, self.env, self.task_suite, self.model, loser_completion_ids)

        # state_chain = execute_action_chain(self.env, processed_units)
        return loser_completion_ids, initial_state

    def get_trajectory_data(self, idx):
        if self.task_num is not None:
            folder_name = self.task_trajectories[str(self.task_num)][idx]
        else:
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
        # start_idx = random.randint(0, len(pkl_files) - 5)
        start_idx = 0
        action_sperate_token_id = 32001
        for i in range(start_idx, len(pkl_files)):
            with open(os.path.join(trajectory_folder_path, pkl_files[i]), "rb") as f:
                data = pickle.load(f)
                state = data["obs"]
                trajectory["state"].append(state)

                action = data["action_ids"].tolist()
                action.append(action_sperate_token_id)
                trajectory["action"].extend(action)

        trajectory["action"] = torch.tensor(trajectory["action"]).unsqueeze(0)
        return trajectory
        # trajectory is a dictionary containing:
        # - "task_num": str, task number extracted from folder name
        # - "episode_num": str, episode number extracted from folder name  
        # - "result": str, either "success" or "failure"
        # - "state": list, will contain observation dictionaries for each step
        # - "action": list, will contain action_ids (tokenized actions) for each step

    def get_initial_state(self, traj, base_vla_name, processor, device, center_crop=True) -> tuple[dict, int]:
        # 返回的inputs是processor处理后的结果, 包括input_ids, attention_mask, pixel_values
        start_idx = random.randint(0, len(traj["state"]) - self.stream_length)
        obs = traj["state"][start_idx]
        task_label = traj["task_description"]
        img = get_libero_image(obs, self.img_size)
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
        return inputs, start_idx

    def get_initial_state_for_loser(self, obs, task_label, base_vla_name, processor, device, center_crop=True)->tuple[dict, int]:
        # 返回的inputs是processor处理后的结果, 包括input_ids, attention_mask, pixel_values
        # obs = traj["state"][start_idx]
        # task_label = traj["task_description"]
        img = get_libero_image(obs, self.img_size)
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

    def execute_action_chain_ids(self, cfg, env, task_suite, model, trajectory, start_idx):
        ACTION_DIM = 7
        task_label = trajectory["task_description"]
        initial_states = task_suite.get_task_init_states(int(trajectory["task_num"]))
        env.reset()
        obs = env.set_init_state(initial_states[int(trajectory["episode_num"])])

        winner_action_chain = trajectory["action"] # [1, seq_len]
        units: List[torch.Tensor] = spilt_chain_to_units(winner_action_chain, cfg.unnorm_key)
        processed_units: List[np.ndarray] = process_action_unit(model, units, cfg.unnorm_key)

        # winner_state_chain = []
        total_length = 0
        action_idx = 0
        while total_length < cfg.num_steps_wait + 550:
            if total_length < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                
                # winner_state_chain.append(obs)
                total_length += 1
                continue
            winner_action = processed_units[action_idx]
            obs, reward, done, info = env.step(winner_action.tolist())
            recorder_state = trajectory["state"][action_idx+1]
            # winner_state_chain.append(obs)
            total_length += 1
            action_idx += 1
            assert action_idx == total_length - cfg.num_steps_wait, f"action_idx {action_idx} is not equal to total_length {total_length} - cfg.num_steps_wait {cfg.num_steps_wait}"

            if total_length >= start_idx:  # begin to get loser action stream
                initial_state = self.get_initial_state_for_loser(obs, task_label, self.cfg.pretrained_checkpoint, self.processor, self.device)

                loser_completion_ids, initial_state = self.get_loser_completion_ids(initial_state) # [1, seq_len]
                loser_units: List[torch.Tensor] = spilt_chain_to_units(loser_completion_ids, cfg.unnorm_key)
                loser_processed_units: List[np.ndarray] = process_action_unit(model, loser_units, cfg.unnorm_key)
                loser_state_chain = []
                for loser_action in loser_processed_units:
                    loser_obs, loser_reward, loser_done, loser_info = env.step(loser_action.tolist())
                    loser_state_chain.append(loser_obs)

                winner_completion_ids = winner_action_chain[:, action_idx * (ACTION_DIM+1) : (action_idx + self.stream_length) * (ACTION_DIM+1)]
                winner_units: List[torch.Tensor] = spilt_chain_to_units(winner_completion_ids, cfg.unnorm_key)
                winner_processed_units: List[np.ndarray] = process_action_unit(model, winner_units, cfg.unnorm_key)
                winner_state_chain = []
                for winner_action in winner_processed_units:
                    winner_obs, winner_reward, winner_done, winner_info = env.step(winner_action.tolist())
                    winner_state_chain.append(winner_obs)
                break
        # print("done:", done)
        return winner_completion_ids, loser_completion_ids, winner_state_chain, loser_state_chain, initial_state


    def get_complete_loser_trajectory(self, traj, start_idx):
        pass
    
    def __len__(self):
        if self.task_num is not None:
            return len(self.task_trajectories[str(self.task_num)])
        else:
            return len(self.trajectory_folders)
    
    def __getitem__(self, idx) -> dict:
        trajectory = self.get_trajectory_data(idx)
        if self.if_offline == False:
            start_idx = random.randint(0, len(trajectory["state"]) - self.stream_length-1)
            start_idx = start_idx + self.cfg.num_steps_wait
            winner_completion_ids, loser_completion_ids, winner_state_chain, loser_state_chain, initial_state = self.execute_action_chain_ids(self.cfg, self.env, self.task_suite, self.model, trajectory, start_idx)
            distance = calculate_average_euclidean_distance(winner_state_chain, loser_state_chain)
            input_ids = initial_state['input_ids']
            pixel_values = initial_state['pixel_values']
            return {"prompt_input_ids":input_ids[0], "pixel_values": pixel_values[0], "chosen_input_ids": winner_completion_ids[0], "rejected_input_ids": loser_completion_ids[0], "distance": distance}
        elif self.if_offline == True:
            # complete_loser_trajectory = self.get_complete_loser_trajectory(trajectory, start_idx)
            complete_winner_trajectory = None
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
        # assert unit[:,-1] == 32001, f"Unit {i} does not end with separator token 32001"
    return units


def process_action_unit(model, units: List[torch.Tensor], unnorm_key) -> List[np.ndarray]:
    """Process action units and convert to continuous actions."""
    processed_units = []
    for unit in units:
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = unit[0, :model.get_action_dim(unnorm_key)].cpu().numpy()
        discretized_actions = model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
        normalized_actions = model.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = model.get_action_stats(unnorm_key)
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



import numpy as np

def calculate_average_euclidean_distance(traj_a, traj_b):
    """
    计算两条轨迹之间的平均欧几里得距离。
    假设两条轨迹长度相同。

    参数:
    traj_a (list): 第一条轨迹，每个元素是 OrderedDict。
    traj_b (list): 第二条轨迹，每个元素是 OrderedDict。

    返回:
    float: 平均距离。
    """
    # 1. 提取末端执行器位置
    positions_a = [step['robot0_eef_pos'] for step in traj_a]
    positions_b = [step['robot0_eef_pos'] for step in traj_b]

    if len(positions_a) != len(positions_b):
        raise ValueError("两条轨迹的长度必须相同才能使用此方法。")

    total_distance = 0.0
    num_steps = len(positions_a)

    # 2. 遍历并计算每一步的距离
    for i in range(num_steps):
        pos_a = positions_a[i]
        pos_b = positions_b[i]
        
        # np.linalg.norm 默认计算L2范数，即欧几里得距离
        distance = np.linalg.norm(pos_a - pos_b)
        total_distance += distance

    # 3. 求平均
    return total_distance / num_steps if num_steps > 0 else 0.0



