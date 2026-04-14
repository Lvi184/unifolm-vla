#!/usr/bin/env python
"""
Script to check image sizes in the datasets
"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# List of all datasets
all_datasets = [
    "clean_the_desktop_addition",
    "clean_the_desktop_part_1",
    "clean_the_desktop_part_2",
    "hold_pot",
    "open_door",
    "place_block_into_box",
    "pour_workpiece",
    "scoop_popcorn",
    "scoop_popcorn_part_2",
    "sorting_packages_part_1",
    "sorting_packages_part_2",
    "sorting_packages_part_3",
    "stock_and_straighten_shelf",
    "stock_and_straighten_shelf_part_2",
    "take_wrong_item_shelf",
]

source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")

print("Checking image sizes for all datasets...\n")

for dataset_name in all_datasets:
    dataset_path = source_dir / dataset_name
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset {dataset_name} not found, skipping...")
        continue
        
    try:
        # Load dataset with larger tolerance
        dataset = LeRobotDataset(repo_id=dataset_name, root=str(dataset_path), 
                                video_backend="pyav", tolerance_s=1.0)
        
        # Get first step
        step = dataset[0]
        
        print(f"📊 {dataset_name}:")
        
        # Check image keys
        image_keys = [k for k in step.keys() if 'image' in k.lower()]
        for img_key in image_keys:
            if img_key in step:
                img = step[img_key]
                if hasattr(img, 'shape'):
                    print(f"  {img_key}: shape={img.shape}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error checking {dataset_name}: {e}")
        print()
