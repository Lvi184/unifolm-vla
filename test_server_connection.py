"""
Test the running UnifoLM-VLA WebSocket server with a simple observation.
"""
import sys
import asyncio
import numpy as np
from PIL import Image

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

from openpi_client import msgpack_numpy
import websockets.asyncio.client as ws_client


async def test_server():
    uri = "ws://localhost:8999"
    
    print("=" * 80)
    print("Testing UnifoLM-VLA WebSocket Server")
    print("=" * 80)
    print(f"Connecting to {uri}...")
    
    try:
        async with ws_client.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Receive metadata
            print("\nReceiving metadata...")
            metadata = msgpack_numpy.unpackb(await websocket.recv())
            print(f"✅ Metadata received:")
            print(f"   Model name: {metadata.get('model_name')}")
            print(f"   Action dim: {metadata.get('action_dim')}")
            print(f"   Checkpoint: {metadata.get('checkpoint')}")
            
            # Create a simple test observation
            print("\nCreating test observation...")
            
            # Dummy image (224x224x3 uint8)
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Dummy state (32D for genie_sim format)
            dummy_state = np.zeros(32, dtype=np.float32)
            
            # Test observation
            observation = {
                "instruction": "Pick up the red block",
                "image_top_head": dummy_image,
                "state": dummy_state
            }
            
            print(f"✅ Test observation created:")
            print(f"   Instruction: {observation['instruction']}")
            print(f"   Image shape: {observation['image_top_head'].shape}")
            print(f"   State shape: {observation['state'].shape}")
            
            # Send observation
            print("\nSending observation to server...")
            packer = msgpack_numpy.Packer()
            await websocket.send(packer.pack(observation))
            print("✅ Observation sent!")
            
            # Receive result
            print("\nWaiting for action prediction...")
            result = msgpack_numpy.unpackb(await websocket.recv())
            
            print("\n" + "=" * 80)
            print("✅ SUCCESS! Received result from server:")
            print("=" * 80)
            
            action = result.get('actions')
            if action is not None:
                print(f"\nAction shape: {action.shape}")
                print(f"Action dtype: {action.dtype}")
                print(f"\nFirst time step (action[0]) - first 10 values:")
                for i, val in enumerate(action[0][:10]):
                    print(f"  [{i:2d}]: {val:.6f}")
            
            if 'server_timing' in result:
                timing = result['server_timing']
                print(f"\nServer timing:")
                if 'infer_ms' in timing:
                    print(f"  Inference time: {timing['infer_ms']:.2f} ms")
                if 'prev_total_ms' in timing:
                    print(f"  Total time: {timing['prev_total_ms']:.2f} ms")
            
            print("\n" + "=" * 80)
            print("Test completed successfully! 🎉")
            print("=" * 80)
            
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_server())
    sys.exit(exit_code)
