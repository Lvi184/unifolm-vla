#!/usr/bin/env python
"""
Check how many episodes are in take_wrong_item_shelf dataset!
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from convert_lerobot_to_hdf5_g2a_resized import LeRobotDataProcessorG2A

dataset_name = "take_wrong_item_shelf"
source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")

source_path = source_dir / dataset_name

print(f"Loading dataset from {source_path}...")
data_processor = LeRobotDataProcessorG2A(
    repo_id=str(source_path), 
    root=None, 
    image_dtype="to_unit8",
    target_size=(400, 640)
)

total_episodes = len(data_processor.dataset.episode_data_index['from'])
print(f"Total episodes in {dataset_name}: {total_episodes}")
