import os
os.environ["HF_HUB_CACHE"] = "/mnt/sda/home/zijianwang/HF_CACHE"
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, EpisodicRLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)
vocab_size = action_tokenizer.vocab_size
print("词表大小:", vocab_size)
vla_model_config = OpenVLAConfig.from_pretrained("openvla/openvla-7b")


batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder if "v01" not in "openvla/openvla-7b" else VicunaV15ChatPromptBuilder,
)

vla_dataset = RLDSDataset(
    "/mnt/sda/home/zijianwang/openvla/modified_libero_rlds",
    "libero_goal_no_noops",
    batch_transform,
    resize_resolution=tuple(vla_model_config.image_sizes),
    shuffle_buffer_size=100_000,
    image_aug=True,
)

episodic_vla_dataset = EpisodicRLDSDataset(
    "/mnt/sda/home/zijianwang/openvla/modified_libero_rlds",
    "libero_goal_no_noops",
    batch_transform,
    resize_resolution=tuple(vla_model_config.image_sizes),
    shuffle_buffer_size=100_000,
    image_aug=False,
    if_random_start=False
)



import re, json
import numpy as np
import pickle
import imageio, os
import numpy as np

task_desc = json.load(open('task_descriptions.json', 'r'))
i = 0
found_tasks = set()  # 用于跟踪已找到的任务
total_tasks = sum(len(tasks) for tasks in task_desc.values())  # 计算总任务数

for data in episodic_vla_dataset:
    print(f"************* Processing episode {i} *************")
    text = processor.decode(data["text"])
    # print(text)
    pattern = r'In:\s*(.*?)\s*Out:'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        # 移除"What action should the robot take to"前缀和末尾的问号
        if extracted_text.startswith("What action should the robot take to "):
            extracted_text = extracted_text[len("What action should the robot take to "):]
        if extracted_text.endswith("?"):
            extracted_text = extracted_text[:-1]
    print(extracted_text)
    # 验证 task_desc 的 value 中是否包含了 extracted_text
    found_match = False
    for task_name, tasks in task_desc.items():
        for task_id, description in tasks.items():
            if extracted_text in description or description in extracted_text:
                print(f"Found match in {task_name}, task {task_id}: {description}")
                found_task_name = task_name
                found_task_id = task_id
                found_match = True
                found_tasks.add((task_name, task_id))  # 记录已找到的任务
                break
        if found_match:
            break
    
    if not found_match:
        print(f"No match found for: {extracted_text}")
    action_ids = data["action"]
    action_ids = np.array(action_ids) 
    imgs = data["replay_images"] #List[np.ndarray]
    # 声明 action_ids 的长度是 imgs 的 8 倍
    assert len(action_ids) == len(imgs) * 8, f"action_ids length {len(action_ids)} should be 8 times imgs length {len(imgs)}"
    # 把action_ids每8个分为1组
    action_ids_grouped = [action_ids[i:i+8] for i in range(0, len(action_ids), 8)]
    trajectory_save_dir = f"winner_trajectory/{found_task_name}/task_{found_task_id}_episode_{-1}/"
    os.makedirs(trajectory_save_dir, exist_ok=True)

    mp4_path = os.path.join(trajectory_save_dir, f"Avideo.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in imgs[:]:    
        video_writer.append_data(img)
    video_writer.close()

    # # Save obs and action_ids with timestep information
    
    for j in range(len(imgs)):
        img = imgs[j]
        info = {
            "obs": img,
            "action_ids": action_ids_grouped[j],
        }
        obs_filename = os.path.join(trajectory_save_dir, f"step_{j}.pkl")
        with open(obs_filename, 'wb') as f:
            pickle.dump(info, f)
    print(f"Saved {obs_filename}")
    i += 1
    
    # 检查终止条件：所有任务都被找到或遍历了1000次
    if len(found_tasks) >= total_tasks or i >= 1000:
        print(f"Stopping loop: found {len(found_tasks)}/{total_tasks} tasks, processed {i} episodes")
        break
    
    # if i > 1000:
    #     break
    # # break
