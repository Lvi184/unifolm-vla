#!/usr/bin/env python
"""
Script to convert all AGIBOT competition datasets from LeRobot format to HDF5 format
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'prepare_data'))

from convert_lerobot_to_hdf5_g2a import lerobot_to_h5_g2a

# List of all AGIBOT competition datasets
AGIBOT_DATASETS = [
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, 
                       default="/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth",
                       help="Source directory containing LeRobot datasets")
    parser.add_argument("--target_dir", type=str,
                       default="/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_hdf5",
                       help="Target directory to save HDF5 datasets")
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {len(AGIBOT_DATASETS)} datasets...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    for dataset_name in AGIBOT_DATASETS:
        source_path = source_dir / dataset_name
        target_path = target_dir / dataset_name
        
        if not source_path.exists():
            print(f"⚠️  Dataset {dataset_name} not found, skipping...")
            continue
            
        print(f"\n⏳ Converting {dataset_name}...")
        
        try:
            lerobot_to_h5_g2a(dataset_name, target_path, str(source_path))
            print(f"✅ Successfully converted {dataset_name}")
        except Exception as e:
            print(f"❌ Error converting {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 All conversions completed!")

if __name__ == "__main__":
    main()
