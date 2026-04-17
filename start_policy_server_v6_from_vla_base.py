"""
UnifoLM-VLA Policy Server (WebSocket - v6)
===============================================

Uses the NEW checkpoint from v3 training (from UnifoLM-VLA-Base)!
"""
import sys
import logging
import traceback
import time
import json
from typing import Any, Union, Tuple, List, Dict

# Add current directory to Python path
sys.path.insert(0, '/root/gpufree-data/unifolm-vla')

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

# Force G2A constants before importing anything else!
sys.argv.append("agibot")

from unifolm_openpi_policy import UnifoLMOpenPIPolicy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as ws_server
import websockets.frames

# Import the verified UnifoLM-VLA components
from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType

import tensorflow as tf
from qwen_vl_utils import process_vision_info
import numpy as np
import torch
from PIL import Image

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
unifolm_vla_IMAGE_SIZE = 224


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =======================================
# Copy of the verified UnifoLM-VLA helpers
# =======================================

def check_image_format(image: Any) -> None:
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        f"Incorrect image format! Need (H, W, 3) uint8, got shape={image.shape}, dtype={image.dtype}"
    )


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def process_image_from_obs(img: Any) -> np.ndarray:
    """
    Process an image from genie_sim observation to UnifoLM-VLA format.
    
    Handles:
    - Torch tensor → numpy array
    - [C, H, W] → [H, W, C]
    - float32 [0,1] → uint8 [0,255]
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # If [C, H, W], convert to [H, W, C]
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # If float in [0,1], convert to uint8 [0,255]
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    
    # Ensure we have [H, W, 3] uint8
    assert len(img.shape) == 3 and img.shape[-1] == 3 and img.dtype == np.uint8, \
        f"Unexpected image format: shape={img.shape}, dtype={img.dtype}"
    
    return img


def unnormalize_action(normalized_actions: np.ndarray, action_norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )
    return actions


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported normalization type!")
    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio


# =======================================
# Load Model (NEW CHECKPOINT!)
# =======================================

CHECKPOINT_PATH = "/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v3_finetune_from_vla_base/checkpoints/steps_8000_pytorch_model.pt"
VLM_PRETRAINED_PATH = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
UNNORM_KEY = "rlds_dataset"
PORT = 8999
HOST = "0.0.0.0"

logger.info("=" * 60)
logger.info(" UnifoLM-VLA Policy Server (v6 - From UnifoLM-VLA-Base!)")
logger.info("=" * 60)
logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
logger.info(f"VLM Pretrained: {VLM_PRETRAINED_PATH}")
logger.info(f"Dataset key: {UNNORM_KEY}")
logger.info(f"Port: {PORT}")
logger.info("")

logger.info("Loading model...")
vla = baseframework.from_pretrained(
    CHECKPOINT_PATH,
    vlm_pretrained_path=VLM_PRETRAINED_PATH
)

logger.info("Converting to bfloat16...")
vla = vla.to(torch.bfloat16)
vla = vla.to(DEVICE).eval()

norm_stats_action = vla.norm_stats[UNNORM_KEY]['action']
norm_stats_proprio = vla.norm_stats[UNNORM_KEY]['proprio']
processor = vla.qwen_vl_interface.processor

# Metadata
max_action = norm_stats_action.get("max", [])
if isinstance(max_action, list):
    action_dim = len(max_action)
else:
    action_dim = getattr(max_action, 'shape', [0])[0]

metadata = {
    "model_name": "unifolm_vla",
    "action_dim": action_dim,
    "checkpoint": CHECKPOINT_PATH,
}

logger.info("Model loaded successfully!")
logger.info(f"Metadata: {metadata}")


# =======================================
# Main Inference Handler
# =======================================

async def handler(websocket: ws_server.ServerConnection):
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()
    
    # Send metadata
    await websocket.send(packer.pack(metadata))
    
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
            
            # =======================================
            # Process observation using VERIFIED logic
            # =======================================
            
            # Handle both single observation and batch observations
            if isinstance(obs, list):
                observations = obs
            else:
                observations = [obs]
                
            # Collect images
            all_images = []
            for observation in observations:
                # Look for images in common places
                found_images = False
                
                # Check 1: observation.images.top_head
                if "observation.images.top_head" in observation:
                    img = process_image_from_obs(observation["observation.images.top_head"])
                    all_images.append(img)
                    found_images = True
                
                # Check 2: image_top_head
                if "image_top_head" in observation:
                    img = process_image_from_obs(observation["image_top_head"])
                    all_images.append(img)
                    found_images = True
                
                # Check 3: full_image
                if "full_image" in observation:
                    img = process_image_from_obs(observation["full_image"])
                    all_images.append(img)
                    found_images = True
                
                # Check 4: images dict (genie_sim format)
                if "images" in observation and isinstance(observation["images"], dict):
                    images_dict = observation["images"]
                    if "top_head" in images_dict:
                        img = process_image_from_obs(images_dict["top_head"])
                        all_images.append(img)
                        found_images = True
                    # Check for wrist images
                    for img_k, img_v in images_dict.items():
                        if "wrist" in img_k.lower() or "hand" in img_k.lower():
                            img = process_image_from_obs(img_v)
                            all_images.append(img)
                
                # Check 5: direct wrist images
                for k, v in observation.items():
                    if isinstance(k, str) and ("wrist" in k.lower() or "hand" in k.lower()):
                        if "observation.images" in k:
                            img = process_image_from_obs(v)
                            all_images.append(img)
                        elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                            img = process_image_from_obs(v)
                            all_images.append(img)
            
            # Get instruction
            instruction = observations[0].get("instruction", "")
            if not instruction and "prompt" in observations[0]:
                instruction = observations[0]["prompt"]
                
            # Log what we found
            logger.info(f"  Found {len(all_images)} images")
            logger.info(f"  Instruction: {instruction}")
            
            # Make sure we have at least one image
            if not all_images:
                logger.warning("No images found in observation! Creating dummy image...")
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                all_images.append(dummy_image)
            
            # =======================================
            # Process images (VERIFIED)
            # =======================================
            processed_images = []
            for image in all_images:
                check_image_format(image)
                if image.shape != (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE, 3):
                    image = resize_image_for_policy(image, unifolm_vla_IMAGE_SIZE)
                pil_image = Image.fromarray(image).convert("RGB")
                processed_images.append(pil_image)
            
            lang = instruction.lower()
            text = f"The task is \"{lang}\"."
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in processed_images],
                        {"type": "text", "text": text},
                    ],
                },
            ]
            
            # =======================================
            # Process inputs (VERIFIED)
            # =======================================
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch_input = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # =======================================
            # Process proprioception (VERIFIED)
            # =======================================
            proprios = []
            for observation in observations:
                if "observation.state" in observation:
                    state = observation["observation.state"]
                elif "state" in observation:
                    state = observation["state"]
                else:
                    # If no state found, use zeros
                    state = np.zeros(159, dtype=np.float32)
                    logger.warning("No state found in observation! Using zeros.")
                
                # Handle different state dimensions
                if len(state) == 32:
                    # GenieSim 32D format: pad to 159D
                    state_padded = np.zeros(159, dtype=np.float32)
                    state_padded[:len(state)] = state
                    state = state_padded
                    logger.info(f"State is 32D, padded to 159D")
                elif len(state) != 159:
                    logger.warning(f"State is {len(state)}D, expected 159D")
                
                proprios.append(state)
                    
            batch_input["state"] = torch.from_numpy(
                normalize_proprio(np.stack(proprios, axis=0), norm_stats_proprio)
            ).unsqueeze(0).to(DEVICE)
            
            # =======================================
            # Move everything to device (VERIFIED)
            # =======================================
            batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
            batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
            batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
            batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)
            
            # =======================================
            # Predict action (VERIFIED - now works!)
            # =======================================
            infer_time = time.monotonic()
            action = vla.predict_action(qwen_inputs=batch_input)
            action = unnormalize_action(action['normalized_actions'][0], norm_stats_action)
            infer_time = time.monotonic() - infer_time
            
            # If we need to convert action back to 32D, do it here
            if len(action) == 40:
                # Take first 32D for genie_sim
                action = action[:32]
                logger.info(f"Action is 40D, sliced to 32D")
            
            # Prepare result
            result = {"actions": action}
            result["server_timing"] = {
                "infer_ms": infer_time * 1000,
            }
            if prev_total_time is not None:
                result["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            
            # Log result
            logger.info(f"Generated action shape: {len(action)}")
            logger.info("=" * 80)
            
            await websocket.send(packer.pack(result))
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
