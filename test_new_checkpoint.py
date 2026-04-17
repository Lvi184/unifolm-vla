#!/usr/bin/env python3
"""
Test script for the new UnifoLM-VLA checkpoint from v3 training.
"""
import sys
import os
import logging
import numpy as np
import torch
from PIL import Image

# Add the project root to path
sys.path.insert(0, '/root/gpufree-data/unifolm-vla')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import UnifoLM-VLA components
from unifolm_vla.model.framework import build_framework
from unifolm_vla.config.training.unifolm_vla_agibot_train import cfg as default_cfg
from omegaconf import OmegaConf

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def load_model(checkpoint_path):
    """
    Load UnifoLM-VLA model with the new checkpoint.
    """
    logger.info(f"Loading config...")
    
    # Load config
    cfg = OmegaConf.load("/root/gpufree-data/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_agibot_train.yaml")
    
    # Override some configs for inference
    cfg.framework.qwenvl.base_vlm = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
    
    logger.info(f"Building model...")
    model = build_framework(cfg)
    model = model.to(DEVICE)
    model.eval()
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load the state dict
    model.load_state_dict(checkpoint, strict=False)
    logger.info("✅ Model loaded successfully!")
    
    return model, cfg

def create_test_input(cfg):
    """
    Create a test input for the model.
    """
    logger.info("Creating test input...")
    
    # Create a dummy image (224x224x3)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Create dummy state (159D)
    dummy_state = np.zeros(159, dtype=np.float32)
    
    # Create dummy text instruction
    dummy_instruction = "Pick up the cube and place it on the shelf."
    
    logger.info(f"Test input created:")
    logger.info(f"  - Image shape: {dummy_image.shape}")
    logger.info(f"  - State shape: {dummy_state.shape}")
    logger.info(f"  - Instruction: {dummy_instruction}")
    
    return {
        "image": pil_image,
        "state": dummy_state,
        "instruction": dummy_instruction
    }

def test_inference(model, cfg, test_input):
    """
    Test model inference.
    """
    logger.info("=" * 60)
    logger.info("Testing inference...")
    logger.info("=" * 60)
    
    from unifolm_openpi_policy import UnifoLMOpenPIPolicy
    
    # Create policy adapter
    logger.info("Creating policy adapter...")
    policy = UnifoLMOpenPIPolicy(
        model_path="/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/checkpoints/steps_8000_pytorch_model.pt",
        base_vlm_path="/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base",
        device="cuda:0"
    )
    
    # Create a test observation similar to genie_sim format
    logger.info("Creating test observation...")
    test_obs = {
        "images": {
            "observation.images.top_head": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        },
        "state": np.zeros(32, dtype=np.float32),  # genie_sim uses 32D state
        "instruction": "Pick up the object"
    }
    
    # Test reset
    logger.info("Testing policy.reset()...")
    policy.reset()
    
    # Test inference
    logger.info("Testing policy.forward()...")
    start_time = time.time()
    
    with torch.no_grad():
        action = policy.forward(test_obs)
    
    inference_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("✅ Inference test successful!")
    logger.info("=" * 60)
    logger.info(f"Inference time: {inference_time:.3f} seconds")
    logger.info(f"Output action shape: {action.shape}")
    logger.info(f"Output action: {action}")
    
    return action

def main():
    """
    Main test function.
    """
    import time
    
    logger.info("=" * 60)
    logger.info("UnifoLM-VLA New Checkpoint Test")
    logger.info("=" * 60)
    
    # Checkpoint path
    checkpoint_path = "/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/checkpoints/steps_8000_pytorch_model.pt"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith(".pt"):
                    logger.info(f"  - {f}")
        return
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Load model and test
    try:
        # First, let's just test the policy adapter directly (simpler)
        logger.info("\n" + "=" * 60)
        logger.info("Testing with UnifoLMOpenPIPolicy...")
        logger.info("=" * 60)
        
        from unifolm_openpi_policy import UnifoLMOpenPIPolicy
        
        # Test all checkpoints
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith("_pytorch_model.pt")])
        
        for ckpt_file in checkpoints:
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
            logger.info(f"\n--- Testing checkpoint: {ckpt_file} ---")
            
            try:
                policy = UnifoLMOpenPIPolicy(
                    model_path=ckpt_path,
                    base_vlm_path="/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base",
                    device="cuda:0"
                )
                
                # Create test observation
                test_obs = {
                    "images": {
                        "observation.images.top_head": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    },
                    "state": np.zeros(32, dtype=np.float32),
                    "instruction": "Pick up the object"
                }
                
                policy.reset()
                
                start_time = time.time()
                action = policy.forward(test_obs)
                inference_time = time.time() - start_time
                
                logger.info(f"✅ {ckpt_file}: Success!")
                logger.info(f"   Inference time: {inference_time:.3f}s")
                logger.info(f"   Action shape: {action.shape}")
                
            except Exception as e:
                logger.error(f"❌ {ckpt_file}: Failed - {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
