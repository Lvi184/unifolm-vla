#!/usr/bin/env python
"""
Continue converting place_block_into_box dataset
"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from convert_lerobot_to_hdf5_g2a import lerobot_to_h5_g2a

dataset_name = "place_block_into_box"
source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")
target_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_hdf5")

print(f"Continuing conversion of {dataset_name}...")

source_path = source_dir / dataset_name
target_path = target_dir / dataset_name

if not source_path.exists():
    print(f"⚠️  Dataset {dataset_name} not found!")
    sys.exit(1)

# Check how many episodes we already have
existing_episodes = list(target_path.glob("episode_*.hdf5"))
existing_episode_ids = [int(f.stem.split("_")[1]) for f in existing_episodes]
print(f"Already converted {len(existing_episode_ids)} episodes")

# Load dataset to see total episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(repo_id=dataset_name, root=str(source_dir), video_backend="pyav", tolerance_s=0.1)
total_episodes = len(dataset.episode_data_index['from'])
print(f"Total episodes in dataset: {total_episodes}")

# Convert missing episodes
for episode_idx in range(total_episodes):
    if episode_idx in existing_episode_ids:
        print(f"Skipping episode {episode_idx} (already exists)")
        continue
        
    print(f"\n⏳ Converting episode {episode_idx}...")
    
    try:
        lerobot_to_h5_g2a(dataset_name, target_path, str(source_path), start_episode=episode_idx, end_episode=episode_idx+1)
        print(f"✅ Successfully converted episode {episode_idx}")
    except Exception as e:
        print(f"❌ Error converting episode {episode_idx}: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 Conversion completed!")
