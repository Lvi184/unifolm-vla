"""
RLDS Dataset Builder for G2A AGIBOT Competition Datasets
Modified for 21D projected data, one HDF5 = one episode
Adds `image_wrist = image_right_wrist` to satisfy UnifoLM camera naming requirement
"""

from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
# Add the directory containing this file to sys.path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths for G2A robot.
    
    One HDF5 file = one full episode, one episode = multiple steps.
    """
    
    def _to_text(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)
    
    def _parse_example(episode_path):
        # Load raw data from converted HDF5 file
        with h5py.File(episode_path, "r") as F:
            actions = F['action'][:]                  # [T, 21]
            proprio = F['observations']["qpos"][:]   # [T, 21] - projected control state
            
            # Images: all RGB cameras from HDF5
            images = F['observations']["images"]
            primary_imgs = images["primary"][:]
            left_wrist_imgs = images["left_wrist"][:] if "left_wrist" in images else None
            right_wrist_imgs = images["right_wrist"][:] if "right_wrist" in images else None
            
            language_raw = F['language_raw'][()].decode('utf-8') if isinstance(F['language_raw'][()], bytes) else str(F['language_raw'][()])
            substep_reasonings = F['substep_reasonings'][:]  # [T] - per-frame instructions
            
            episode_length = actions.shape[0]

            # Validate dimensions before yielding
            assert actions.ndim == 2 and actions.shape[1] == 21, \
                f"Expected actions shape [T, 21], got {actions.shape}"
            assert proprio.ndim == 2 and proprio.shape[1] == 21, \
                f"Expected proprio shape [T, 21], got {proprio.shape}"
            assert primary_imgs.shape[0] == episode_length, \
                f"Primary images {primary_imgs.shape[0]} != episode_length {episode_length}"
            if left_wrist_imgs is not None:
                assert left_wrist_imgs.shape[0] == episode_length, \
                    f"Left wrist images {left_wrist_imgs.shape[0]} != episode_length {episode_length}"
            if right_wrist_imgs is not None:
                assert right_wrist_imgs.shape[0] == episode_length, \
                    f"Right wrist images {right_wrist_imgs.shape[0]} != episode_length {episode_length}"

        # Yield the entire episode as one RLDS episode with T steps
        yield f"{str(episode_path).replace('/', '_')}", {
            'steps': [
                {
                    'observation': {
                        'image_primary': primary_imgs[i],
                        'image_left_wrist': left_wrist_imgs[i] if left_wrist_imgs is not None else None,
                        'image_right_wrist': right_wrist_imgs[i] if right_wrist_imgs is not None else None,
                        'image_wrist': right_wrist_imgs[i] if right_wrist_imgs is not None else None,
                        'proprio': proprio[i],
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'is_first': i == 0,
                    'is_last': i == (episode_length - 1),
                    'is_terminal': i == (episode_length - 1),
                    'language_instruction': _to_text(substep_reasonings[i]),
                }
                for i in range(episode_length)
            ],
            'episode_metadata': {
                'file_path': str(episode_path),
            },
        }

    # For smallish datasets, use single-threaded parsing
    for sample in paths:
        yield from _parse_example(sample)


class rlds_dataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for G2A AGIBOT competition datasets (21D projected)."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release for G2A AGIBOT competition 21D projected dataset.',
    }
    N_WORKERS = 8            # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 8  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata for projected 21D G2A robot."""
        # Actual image dimensions from agibot dataset:
        # - primary (top_head): 400 x 640 x 3
        # - left/right wrist: 1056 x 1280 x 3
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_primary': tfds.features.Image(
                            shape=(400, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Top head camera (primary) RGB observation.',
                        ),
                        'image_left_wrist': tfds.features.Image(
                            shape=(1056, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left wrist camera RGB observation.',
                        ),
                        'image_right_wrist': tfds.features.Image(
                            shape=(1056, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'image_wrist': tfds.features.Image(
                            shape=(1056, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera (alias for UnifoLM compatibility) RGB observation.',
                        ),
                        'proprio': tfds.features.Tensor(
                            shape=(21,),
                            dtype=np.float32,
                            doc='Projected G2A robot control proprioception (21D, from 159D).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(21,),
                        dtype=np.float32,
                        doc='Projected G2A robot control action (21D, from 40D).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Per-frame language instruction (from instruction_segments).'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({ 
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))
        
    def _split_paths(self):
        """Define filepaths for 21D G2A data splits."""
        # Get all HDF5 files from all converted task directories
        hdf5_files = []
        base_dir = os.environ.get("UNIFOLM_HDF5_ROOT", "/root/gpufree-data/unifolm_hdf5")
        
        # Collect all HDF5 files from all task directories
        if os.path.exists(base_dir):
            for task_dir in os.listdir(base_dir):
                task_path = os.path.join(base_dir, task_dir)
                if os.path.isdir(task_path):
                    hdf5_pattern = os.path.join(task_path, "*.hdf5")
                    hdf5_files.extend(glob.glob(hdf5_pattern))
        
        print(f"Found {len(hdf5_files)} HDF5 files for 21D G2A datasets")
        return {
            'train': hdf5_files,
        }
