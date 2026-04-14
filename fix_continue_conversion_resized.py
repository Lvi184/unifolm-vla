
#!/usr/bin/env python
"""
Continue converting place_block_into_box dataset with resized images!
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data'))

from convert_lerobot_to_hdf5_g2a_resized import LeRobotDataProcessorG2A, H5Writer

dataset_name = "place_block_into_box"
source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")
target_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_hdf5")

print(f"Continuing conversion of {dataset_name} with resized images (400x640)...")

source_path = source_dir / dataset_name
target_path = target_dir / dataset_name

if not source_path.exists():
    print(f"⚠️  Dataset {dataset_name} not found!")
    sys.exit(1)

# Check how many episodes we already have
existing_episodes = list(target_path.glob("episode_*.hdf5"))
existing_episode_ids = [int(f.stem.split("_")[1]) for f in existing_episodes]
print(f"Already converted {len(existing_episode_ids)} episodes")

# Load dataset - use full path as repo_id to avoid HuggingFace connection!
print(f"Loading dataset from {source_path}...")
data_processor = LeRobotDataProcessorG2A(
    repo_id=str(source_path), 
    root=None, 
    image_dtype="to_unit8",
    target_size=(400, 640)
)

total_episodes = len(data_processor.dataset.episode_data_index['from'])
print(f"Total episodes in dataset: {total_episodes}")

h5_writer = H5Writer(target_path)

# Convert missing episodes
for episode_idx in range(total_episodes):
    if episode_idx in existing_episode_ids:
        print(f"Skipping episode {episode_idx} (already exists)")
        continue
        
    print(f"\n⏳ Converting episode {episode_idx}...")
    
    try:
        episode = data_processor.process_episode(episode_idx)
        h5_writer.write_to_h5(episode)
        print(f"✅ Successfully converted episode {episode_idx}")
    except Exception as e:
        print(f"❌ Error converting episode {episode_idx}: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 Conversion completed!")
