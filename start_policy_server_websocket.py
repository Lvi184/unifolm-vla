"""
UnifoLM-VLA Policy Server (WebSocket - OpenPI compatible)
=====================================================

This is the correct version for genie_sim! Uses WebSocket protocol
with msgpack-numpy encoding, same as ACoT-VLA.

Usage:
    python start_policy_server_websocket.py
"""
import sys
import logging
import time
from typing import Dict

import numpy as np
import torch
import asyncio
import websockets.asyncio.server as ws_server
import websockets.frames

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

from openpi_client import msgpack_numpy

# Force G2A constants before importing anything else!
sys.argv.append("agibot")

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
logger.info("=" * 60)
logger.info(" UnifoLM-VLA Policy Server (WebSocket - OpenPI compatible)")
logger.info("=" * 60)
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
logger.info("")


async def handler(websocket: ws_server.ServerConnection):
    """WebSocket connection handler."""
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()
    
    # Send metadata first
    await websocket.send(packer.pack(policy.metadata))
    
    prev_total_time = None
    
    while True:
        try:
            start_time = time.monotonic()
            
            # Receive observation
            obs = msgpack_numpy.unpackb(await websocket.recv())
            
            infer_time = time.monotonic()
            action = policy.infer(obs)
            infer_time = time.monotonic() - infer_time
            
            action["server_timing"] = {
                "infer_ms": infer_time * 1000,
            }
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            
            await websocket.send(packer.pack(action))
            prev_total_time = time.monotonic() - start_time
            
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
            break
        except Exception:
            import traceback
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise


async def health_check(connection, request):
    """Health check endpoint for HTTP."""
    if request.path == "/healthz":
        return connection.respond(200, "OK\n")
    return None


async def main():
    logger.info(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    logger.info(f"Health check: http://{HOST}:{PORT}/healthz")
    logger.info("Press Ctrl+C to stop")
    
    async with ws_server.serve(
        handler,
        HOST,
        PORT,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
