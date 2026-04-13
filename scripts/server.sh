#!/bin/bash
# UnifoLM-VLA server startup script
# Similar to ACoT-VLA's server.sh

cart_num=${1}
port=${2}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/app:/app/src
export GIT_LFS_SKIP_SMUDGE=1

# Start UnifoLM-VLA WebSocket server
cd /app
python start_policy_server_final_v5.py
