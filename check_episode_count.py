#!/usr/bin/env python
"""Check number of episodes in a dataset"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_name = "place_block_into_box"
source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")

print(f"Loading dataset: {dataset_name}")
dataset = LeRobotDataset(repo_id=dataset_name, root=str(source_dir), video_backend="pyav", tolerance_s=0.1)

print(f"Number of episodes: {len(dataset.episode_data_index['from'])}")
print(f"Total steps: {len(dataset)}")
