"""
RLDS Dataset Builder for G2A AGIBOT Competition Datasets
Modified from rlds_dataset.py for G2A robot data
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
    """Yields episodes for list of data paths for G2A robot."""
    
    def _parse_example(episode_path):
        # Load raw data from G2A HDF5 file
        with h5py.File(episode_path, "r") as F:
            actions = F['action'][:]
            states = F['observations']["qpos"][:]
            ee_states = F['observations']["ee_qpos"][:]
            ee_actions = F["ee_action"][:]
            
            # G2A images - top_head, hand_left, hand_right
            images_top_head = F['observations']["images"]["top_head"][:]  
            images_hand_left = F['observations']["images"]["hand_left"][:]  
            images_hand_right = F['observations']["images"]["hand_right"][:]  
            
            language_instruction = F['language_raw'][()].decode('utf-8') if isinstance(F['language_raw'][()], bytes) else str(F['language_raw'][()])
            
            episode_length = actions.shape[0]

        # Yield each step
        for i in range(episode_length):
            yield f"{episode_path.replace('/', '_')}_{i}", {
                'steps': [{
                    'observation': {
                        'image_top_head': images_top_head[i],
                        'image_hand_left': images_hand_left[i],
                        'image_hand_right': images_hand_right[i],
                        'state': states[i],
                        'ee_state': ee_states[i],
                    },
                    'action': actions[i],
                    'ee_action': ee_actions[i],
                    'discount': 1.0,
                    'is_first': i == 0,
                    'is_last': i == (episode_length - 1),
                    'is_terminal': i == (episode_length - 1),
                    'language_instruction': language_instruction,
                }],
                'episode_metadata': {
                    'file_path': episode_path,
                }
            }

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        yield from _parse_example(sample)


class rlds_dataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for G2A AGIBOT competition datasets."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release for G2A AGIBOT competition datasets.',
    }
    N_WORKERS = 8            # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 8  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata for G2A robot."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_top_head': tfds.features.Image(
                            shape=(400, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Top head camera RGB observation for G2A robot.',
                        ),
                        'image_hand_left': tfds.features.Image(
                            shape=(400, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left hand camera RGB observation for G2A robot.',
                        ),
                        'image_hand_right': tfds.features.Image(
                            shape=(400, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right hand camera RGB observation for G2A robot.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(159,),
                            dtype=np.float32,
                            doc='G2A robot joint state (159D).',
                        ),
                        'ee_state': tfds.features.Tensor(
                            shape=(159,),
                            dtype=np.float32,
                            doc='G2A robot end effector state (159D).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(40,),
                        dtype=np.float32,
                        doc='G2A robot joint action (40D).',
                    ),
                    'ee_action': tfds.features.Tensor(
                        shape=(40,),
                        dtype=np.float32,
                        doc='G2A robot end effector action (40D).',
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
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({ 
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))
        
        
        
    def _split_paths(self):
        """Define filepaths for G2A data splits."""
        # Get all HDF5 files from all G2A datasets
        hdf5_files = []
        base_dir = "/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_hdf5"
        
        # Collect all HDF5 files from all dataset directories
        if os.path.exists(base_dir):
            for dataset_dir in os.listdir(base_dir):
                dataset_path = os.path.join(base_dir, dataset_dir)
                if os.path.isdir(dataset_path):
                    hdf5_pattern = os.path.join(dataset_path, "*.hdf5")
                    hdf5_files.extend(glob.glob(hdf5_pattern))
        
        print(f"Found {len(hdf5_files)} HDF5 files for G2A datasets")
        return {
            'train': hdf5_files,
        }
