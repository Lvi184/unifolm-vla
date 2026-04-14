"""
Final UnifoLM-VLA Policy Server (WebSocket - OpenPI compatible)
=====================================================

Uses the updated unifolm_openpi_policy.py with:
- Proper image handling from obs["images"] dict
- Proper format conversion (C,H,W → H,W,C; float32 → uint8)
- Dimension conversion between 32D ↔ 159D/40D
"""
import sys
import logging
import traceback
import time  # Import at top for handler

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

# Force G2A constants before importing anything else!
sys.argv.append("agibot")

from unifolm_openpi_policy import UnifoLMOpenPIPolicy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as ws_server
import websockets.frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# Configuration
CHECKPOINT_PATH = "./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt"
VLM_PRETRAINED_PATH = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
UNNORM_KEY = "rlds_dataset"
PORT = 8999
HOST = "0.0.0.0"

# Load policy
logger.info("=" * 60)
logger.info(" UnifoLM-VLA Policy Server (FINAL v3)")
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


async def handler(websocket: ws_server.ServerConnection):
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()
    
    # Send metadata
    await websocket.send(packer.pack(policy.metadata))
    
    prev_total_time = None
    
    while True:
        try:
            start_time = time.monotonic()
            
            # Receive observation
            obs = msgpack_numpy.unpackb(await websocket.recv())
            
            # Log observation structure
            logger.info("=" * 80)
            logger.info("Received observation:")
            logger.info(f"  Keys: {list(obs.keys())}")
            
            if "state" in obs:
                state = obs["state"]
                logger.info(f"  State shape: {state.shape if hasattr(state, 'shape') else len(state)}")
                logger.info(f"  State dtype: {state.dtype if hasattr(state, 'dtype') else type(state)}")
                if hasattr(state, '__len__') and len(state) <= 50:
                    logger.info(f"  State full: {state}")
                elif hasattr(state, '__len__'):
                    logger.info(f"  State sample (first 30): {state[:30]}")
            
            for k in ["prompt", "instruction", "language_instruction", "task_name"]:
                if k in obs:
                    logger.info(f"  {k}: {obs[k]}")
            
            # Process and log images if needed
            if "images" in obs and isinstance(obs["images"], dict):
                logger.info(f"  Images dict keys: {list(obs['images'].keys())}")
                for img_k, img_v in obs["images"].items():
                    if hasattr(img_v, 'shape'):
                        logger.info(f"    {img_k}: shape={img_v.shape}, dtype={img_v.dtype}")
                    else:
                        logger.info(f"    {img_k}: type={type(img_v)}")
            
            infer_time = time.monotonic()
            action = policy.infer(obs)
            infer_time = time.monotonic() - infer_time
            
            action["server_timing"] = {
                "infer_ms": infer_time * 1000,
            }
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            
            # Log action
            logger.info(f"Generated action shape: {action['actions'].shape if hasattr(action['actions'], 'shape') else len(action['actions'])}")
            logger.info("=" * 80)
            
            await websocket.send(packer.pack(action))
            prev_total_time = time.monotonic() - start_time
            
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
            break
        except Exception as e:
            logger.error(f"Error: {traceback.format_exc()}")
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
    import asyncio
    asyncio.run(main())
