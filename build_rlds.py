#!/usr/bin/env python3
"""
Build RLDS dataset from resized HDF5 files!
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data/hdf5_to_rlds'))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tensorflow_datasets as tfds
from rlds_dataset.rlds_dataset import rlds_dataset

print("Starting RLDS dataset build...")

data_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_rlds")

builder = rlds_dataset(data_dir=str(data_dir))

# Disable GCS check
download_config = tfds.download.DownloadConfig(
    register_checksums=False,
    max_examples_per_split=None,
)

builder.download_and_prepare(download_config=download_config)

print("🎉 RLDS dataset build completed!")
