
#!/usr/bin/env python
"""Test loading local LeRobot dataset"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_name = "place_block_into_box"
source_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth")

print(f"Loading dataset {dataset_name} from {source_dir}")

# Try different ways to load the dataset
try:
    # Method 1: repo_id=directory name, root=parent directory
    print("\nMethod 1: repo_id=dataset_name, root=source_dir")
    dataset1 = LeRobotDataset(repo_id=dataset_name, root=str(source_dir), video_backend="pyav", tolerance_s=0.1)
    print(f"✅ Success! Number of episodes: {len(dataset1.episode_data_index['from'])}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

try:
    # Method 2: repo_id=full path?
    print("\nMethod 2: repo_id=full path")
    dataset2 = LeRobotDataset(repo_id=str(source_dir / dataset_name), video_backend="pyav", tolerance_s=0.1)
    print(f"✅ Success! Number of episodes: {len(dataset2.episode_data_index['from'])}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
