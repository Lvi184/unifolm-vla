#!/usr/bin/env python
"""Check number of episodes in a dataset"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

datasets_to_check = [
    "clean_the_desktop_addition",
    "take_wrong_item_shelf",
    "place_block_into_box",
]

source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")

for dataset_name in datasets_to_check:
    print(f"\n=== Checking {dataset_name} ===")
    try:
        dataset = LeRobotDataset(repo_id=dataset_name, root=str(source_dir), video_backend="pyav", tolerance_s=0.5)
        print(f"  Number of episodes: {len(dataset.episode_data_index['from'])}")
        print(f"  Total steps: {len(dataset)}")
    except Exception as e:
        print(f"  Error: {e}")
