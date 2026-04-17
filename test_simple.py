#!/usr/bin/env python3
"""
Simple test for the new UnifoLM-VLA checkpoint.
"""
import sys
import os
import time
import logging
import numpy as np

# Force G2A constants before importing anything else!
sys.argv = [sys.argv[0], "agibot"]

# Add the project root to path
sys.path.insert(0, '/root/gpufree-data/unifolm-vla')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_checkpoint(ckpt_name, ckpt_path):
    """Test a single checkpoint."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {ckpt_name}")
    logger.info(f"{'='*60}")
    
    try:
        from unifolm_openpi_policy import UnifoLMOpenPIPolicy
        
        # Load policy
        logger.info(f"Loading policy from: {ckpt_path}")
        start_load = time.time()
        
        policy = UnifoLMOpenPIPolicy(
            ckpt_path=ckpt_path,
            vlm_pretrained_path="/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
        )
        
        load_time = time.time() - start_load
        logger.info(f"✅ Loaded in {load_time:.2f}s")
        
        # Create test observation
        logger.info("Creating test observation...")
        test_obs = {
            "images": {
                "observation.images.top_head": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            },
            "state": np.zeros(32, dtype=np.float32),
            "instruction": "Pick up the cube and place it on the shelf."
        }
        
        # Test multiple inferences
        logger.info("Running inference tests...")
        inference_times = []
        
        for i in range(3):
            start_infer = time.time()
            result = policy.infer(test_obs)
            action = result["actions"]
            infer_time = time.time() - start_infer
            inference_times.append(infer_time)
            
            logger.info(f"  Inference {i+1}: {infer_time:.3f}s, action shape={action.shape}")
        
        avg_infer_time = np.mean(inference_times)
        logger.info(f"✅ Average inference time: {avg_infer_time:.3f}s")
        
        return True, avg_infer_time
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main test function."""
    logger.info("="*60)
    logger.info("UnifoLM-VLA Checkpoint Test")
    logger.info("="*60)
    
    # Checkpoint directory
    ckpt_dir = "/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/checkpoints"
    
    if not os.path.exists(ckpt_dir):
        logger.error(f"❌ Checkpoint directory not found: {ckpt_dir}")
        return
    
    # Get all checkpoints
    checkpoints = []
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith("_pytorch_model.pt"):
            checkpoints.append(f)
    
    if not checkpoints:
        logger.error(f"❌ No checkpoints found in: {ckpt_dir}")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        logger.info(f"  - {ckpt}")
    
    # Test each checkpoint
    results = {}
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        success, infer_time = test_checkpoint(ckpt_name, ckpt_path)
        results[ckpt_name] = (success, infer_time)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    for ckpt_name, (success, infer_time) in results.items():
        if success:
            logger.info(f"✅ {ckpt_name}: OK (avg {infer_time:.3f}s)")
        else:
            logger.info(f"❌ {ckpt_name}: FAILED")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()
