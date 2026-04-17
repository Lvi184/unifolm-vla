#!/usr/bin/env python3
"""
UnifoLM-VLA Policy Server (v3 Checkpoint)
=========================================

WebSocket server that serves the UnifoLM-VLA policy for genie_sim,
with FIXED 21D dimension conversion (EXACTLY matching official ACoT-VLA!).

Uses v3 checkpoint: /root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/checkpoints/steps_8000_pytorch_model.pt
"""

import sys
import asyncio
import logging
import numpy as np
from aiohttp import web, WSMsgType

# Add current directory to Python path
sys.path.insert(0, '/root/gpufree-data/unifolm-vla')

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

# Force G2A constants before importing anything else!
sys.argv.append("agibot")

from openpi_client import msgpack_numpy
import msgpack

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our policy adapter
from unifolm_openpi_policy import UnifoLMOpenPIPolicy

# Global policy instance
policy = None

async def health_check(request):
    """Health check endpoint."""
    return web.json_response({"status": "ok", "policy": "unifolm_vla_v3"})

async def websocket_handler(request):
    """Handle WebSocket connections."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info("New WebSocket connection established")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                try:
                    # Unpack the message
                    data = msgpack.unpackb(msg.data, raw=False)
                    
                    logger.info(f"Received request: {type(data)}")
                    if isinstance(data, dict):
                        logger.info(f"  Keys: {list(data.keys())}")
                        if 'images' in data:
                            logger.info(f"  Images keys: {list(data['images'].keys())}")
                        if 'state' in data:
                            logger.info(f"  State shape: {np.array(data['state']).shape}")
                    
                    # Run inference with our policy
                    if policy is not None:
                        result = policy.infer(data)
                        logger.info(f"Inference result: actions shape={np.array(result['actions']).shape}")
                        
                        # Pack and send the result
                        result_data = msgpack.packb(result)
                        await ws.send_bytes(result_data)
                    else:
                        logger.error("Policy not initialized!")
                        error_result = {"error": "Policy not initialized"}
                        await ws.send_bytes(msgpack.packb(error_result))
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    error_result = {"error": str(e)}
                    await ws.send_bytes(msgpack.packb(error_result))
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed")
    
    return ws

def init_policy():
    """Initialize the UnifoLM-VLA policy."""
    global policy
    
    logger.info("=" * 80)
    logger.info("Initializing UnifoLM-VLA Policy (v3 checkpoint)...")
    logger.info("=" * 80)
    
    # Paths
    ckpt_path = "/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/checkpoints/steps_8000_pytorch_model.pt"
    vlm_path = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
    
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"VLM: {vlm_path}")
    logger.info("")
    
    # Initialize policy
    policy = UnifoLMOpenPIPolicy(
        ckpt_path=ckpt_path,
        vlm_pretrained_path=vlm_path,
        unnorm_key="rlds_dataset",
        center_crop=False,
        use_bf16=True
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ UnifoLM-VLA Policy (v3) initialized successfully!")
    logger.info("=" * 80)

def main():
    """Main entry point."""
    # Initialize policy first
    init_policy()
    
    # Create web app
    app = web.Application()
    
    # Add routes
    app.router.add_get('/healthz', health_check)
    app.router.add_get('/', websocket_handler)
    
    # Start server
    host = "0.0.0.0"
    port = 8999
    
    logger.info("")
    logger.info("🚀 Starting UnifoLM-VLA WebSocket Server (v3)...")
    logger.info(f"📍 WebSocket: ws://{host}:{port}")
    logger.info(f"📍 Health check: http://{host}:{port}/healthz")
    logger.info("")
    logger.info("Ready to accept connections!")
    logger.info("")
    
    web.run_app(app, host=host, port=port)

if __name__ == "__main__":
    main()
