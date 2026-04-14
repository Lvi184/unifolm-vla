#!/usr/bin/env python
"""
Script to check the structure of the AGIBOT competition datasets
"""
import os
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    dataset_name = "clean_the_desktop_addition"
    dataset_path = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth") / dataset_name
    
    print(f"Checking dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    
    try:
        dataset = LeRobotDataset(repo_id=dataset_name, root=str(dataset_path), video_backend="pyav")
        print(f"\n✅ Dataset loaded successfully!")
        print(f"Number of episodes: {dataset.num_episodes}")
        
        # Get the first step
        print(f"\n📊 First step data structure:")
        step = dataset[0]
        for key in sorted(step.keys()):
            value = step[key]
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}, value={value}")
        
        # Get episode index
        print(f"\n📊 Episode data index:")
        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()
        print(f"  Episode 0: from={from_idx}, to={to_idx}, length={to_idx-from_idx}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
