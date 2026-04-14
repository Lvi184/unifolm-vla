#!/usr/bin/env python
"""
Test script to verify model loading from existing checkpoint
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from unifolm_vla.model.framework.base_framework import baseframework

print("Testing model loading...")

# Paths
ckpt_path = "/root/gpufree-data/unifolm-weights/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt"
vlm_path = "/root/gpufree-data/unifolm-weights/UnifoLM-VLA-Base"

print(f"Checkpoint path: {ckpt_path}")
print(f"VLM path: {vlm_path}")

try:
    # Try loading the model
    print("Loading model...")
    model = baseframework.from_pretrained(
        ckpt_path,
        vlm_pretrained_path=vlm_path
    )
    
    print("✅ Model loaded successfully!")
    print(f"Model has norm_stats keys: {list(model.norm_stats.keys())}")
    
    # Move model to CUDA
    model = model.to("cuda").eval()
    print("✅ Model moved to CUDA")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
