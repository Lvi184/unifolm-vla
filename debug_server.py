"""
Debug server to inspect what genie_sim actually sends.
"""
import sys
import logging
import traceback

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

from openpi_client import msgpack_numpy
import websockets.asyncio.server as ws_server
import websockets.frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def handler(websocket: ws_server.ServerConnection):
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()
    
    # Send fake metadata
    metadata = {"model_name": "debug", "action_dim": 40}
    await websocket.send(packer.pack(metadata))
    
    while True:
        try:
            # Receive observation
            obs = msgpack_numpy.unpackb(await websocket.recv())
            
            logger.info("=" * 80)
            logger.info("Received observation:")
            logger.info(f"  Keys: {list(obs.keys())}")
            
            if "observation" in obs:
                logger.info(f"  Observation keys: {list(obs['observation'].keys())}")
                
                # Check state
                if "state" in obs["observation"]:
                    state = obs["observation"]["state"]
                    logger.info(f"  State shape: {state.shape}")
                    logger.info(f"  State dtype: {state.dtype}")
                    logger.info(f"  State sample (first 20): {state[:20]}")
                
                # Check images
                for k, v in obs["observation"].items():
                    if "image" in k.lower():
                        if hasattr(v, 'shape'):
                            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                        else:
                            logger.info(f"  {k}: type={type(v)}")
            
            # Check for instruction/prompt
            for k in ["prompt", "instruction", "language_instruction"]:
                if k in obs:
                    logger.info(f"  {k}: {obs[k]}")
            
            # Send fake action
            import numpy as np
            fake_action = np.zeros(40, dtype=np.float32)
            result = {"actions": fake_action}
            await websocket.send(packer.pack(result))
            
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


async def main():
    logger.info("Starting debug server on ws://0.0.0.0:8999")
    async with ws_server.serve(
        handler,
        "0.0.0.0",
        8999,
        compression=None,
        max_size=None,
    ) as server:
        await server.serve_forever()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
