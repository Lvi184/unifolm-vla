#!/usr/bin/env python3
"""
Build RLDS dataset from resized HDF5 files (OFFLINE MODE)!
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / 'prepare_data/hdf5_to_rlds'))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow_datasets as tfds
from rlds_dataset.rlds_dataset import rlds_dataset

print("="*80)
print("Starting OFFLINE RLDS dataset build...")
print("="*80)

data_dir = Path("/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_rlds")

print(f"Data directory: {data_dir}")

# Initialize builder
builder = rlds_dataset(data_dir=str(data_dir))

# Configure download for OFFLINE mode
download_config = tfds.download.DownloadConfig(
    register_checksums=False,
    force_checksums_validation=False,
    max_examples_per_split=None,
    beam_options=None,
    beam_runner=None,
)

print("Starting download_and_prepare()...")
builder.download_and_prepare(download_config=download_config)

print("="*80)
print("🎉 OFFLINE RLDS dataset build completed!")
print("="*80)
