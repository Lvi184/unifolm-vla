#!/usr/bin/env python
"""
Script to convert the missing datasets to HDF5 format
"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from convert_lerobot_to_hdf5_g2a import lerobot_to_h5_g2a

# Missing datasets
missing_datasets = [
    "take_wrong_item_shelf",
    "place_block_into_box",
]

source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")
target_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_hdf5")

print(f"Converting {len(missing_datasets)} missing datasets...")

for dataset_name in missing_datasets:
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

print("\n🎉 All missing datasets conversion completed!")
