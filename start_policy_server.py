"""
Simple UnifoLM-VLA Policy Server (FastAPI)
=========================================

This is the most stable version - uses FastAPI and explicitly
sets G2A constants (for AGIBOT dataset compatibility).

Usage:
    python start_policy_server.py
"""
import sys
import logging
import time
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Force G2A constants before importing anything else!
sys.argv.append("agibot")

# Now import everything else
from unifolm_openpi_policy import UnifoLMOpenPIPolicy

# Configuration
CHECKPOINT_PATH = "./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt"
VLM_PRETRAINED_PATH = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
UNNORM_KEY = "rlds_dataset"
PORT = 8999
HOST = "0.0.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Load policy
logger.info("=" * 50)
logger.info(" UnifoLM-VLA Policy Server (FastAPI)")
logger.info("=" * 50)
logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
logger.info(f"VLM Pretrained: {VLM_PRETRAINED_PATH}")
logger.info(f"Dataset key: {UNNORM_KEY}")
logger.info(f"Port: {PORT}")
logger.info("")

policy = UnifoLMOpenPIPolicy(
    ckpt_path=CHECKPOINT_PATH,
    vlm_pretrained_path=VLM_PRETRAINED_PATH,
    unnorm_key=UNNORM_KEY,
)

logger.info("")
logger.info("Policy loaded successfully!")
logger.info(f"Metadata: {policy.metadata}")

# Create FastAPI app
app = FastAPI(
    title="UnifoLM-VLA Policy Server",
    description="Vision-Language-Action Model Inference",
    version="1.0.0",
)

@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok", "model": "unifolm_vla"}

@app.get("/metadata")
async def metadata():
    """Get policy metadata."""
    return policy.metadata

@app.post("/act")
async def act(obs: Dict):
    """
    Main inference endpoint.
    
    Accepts LeRobot/genie_sim format observations like:
    {
        "observation.images.top_head": np.array([H, W, 3], dtype=np.uint8),
        "observation.state": np.array([159], dtype=np.float32),
        "prompt": "clean the table",
    }
    
    Returns:
    {
        "actions": np.array([40], dtype=np.float32),
    }
    """
    try:
        t1 = time.time()
        result = policy.infer(obs)
        inference_time = time.time() - t1
        logger.info(f"Inference: {inference_time*1000:.1f}ms")
        return JSONResponse(result)
    except Exception as e:
        import traceback
        logger.error(f"Error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    logger.info(f"Starting server on http://{HOST}:{PORT}")
    logger.info(f"Health check: http://{HOST}:{PORT}/healthz")
    logger.info(f"Metadata: http://{HOST}:{PORT}/metadata")
    logger.info(f"Inference: POST http://{HOST}:{PORT}/act")
    logger.info("Press Ctrl+C to stop")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
