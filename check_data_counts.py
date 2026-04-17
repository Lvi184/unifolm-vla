
#!/usr/bin/env python3
"""
Check episode counts in original vs converted data.
"""

import json
from pathlib import Path

original_base = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")
hdf5_base = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_hdf5")

datasets = [
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

print("=" * 80)
print("CHECKING DATASET EPISODE COUNTS")
print("=" * 80)
print()

total_original = 0
total_hdf5 = 0

print(f"{'Dataset':<40} {'Original':<10} {'HDF5':<10} {'Match?'}")
print("-" * 80)

for dataset in datasets:
    orig_path = original_base / dataset / "meta" / "info.json"
    hdf5_path = hdf5_base / dataset
    
    orig_count = 0
    hdf5_count = 0
    
    if orig_path.exists():
        with open(orig_path, 'r') as f:
            orig_info = json.load(f)
            orig_count = orig_info.get('total_episodes', 0)
    
    if hdf5_path.exists():
        hdf5_files = list(hdf5_path.glob("*.hdf5"))
        hdf5_count = len(hdf5_files)
    
    match = "✓" if orig_count == hdf5_count else "✗"
    print(f"{dataset:<40} {orig_count:<10} {hdf5_count:<10} {match}")
    
    total_original += orig_count
    total_hdf5 += hdf5_count

print("-" * 80)
print(f"{'TOTAL':<40} {total_original:<10} {total_hdf5:<10} {'✓' if total_original == total_hdf5 else '✗'}")
print()

if total_original != total_hdf5:
    print("❌ DATA COUNT MISMATCH! Possible data loss during conversion!")
else:
    print("✅ Data counts match!")

print()
