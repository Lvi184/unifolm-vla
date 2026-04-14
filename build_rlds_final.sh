#!/bin/bash
# Build RLDS dataset from HDF5 files!
set -e

cd /root/gpufree-data/unifolm-vla/prepare_data/hdf5_to_rlds

# Activate virtual environment
source /root/gpufree-data/unifolm-vla/.venv/bin/activate

# Disable GCS checks
export TFDS_DATA_DIR=/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_rlds
export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting RLDS build..."

# Go to the rlds_dataset directory and run tfds build
cd rlds_dataset
tfds build --overwrite --data_dir="$TFDS_DATA_DIR"

echo "✅ RLDS build completed!"
