#!/usr/bin/env python3
"""
UnifoLM-VLA Policy Server (ACoT 32D Direct version)
=============================================================

Direct 32D input/output matching genie_sim competition interface.
No projection, no remapping. Fully aligned with 32D training.
"""

import sys
import os
import logging
import traceback
import time
from typing import Any, Union, Tuple, List, Dict

import msgpack_numpy

# Add current directory to Python path
sys.path.insert(0, '/root/gpufree-data/unifolm-vla')

# Add genie_sim openpi-client to path
sys.path.insert(0, '/root/genie_sim/openpi/packages/openpi-client/src')

# Force G2A constants before importing anything else!
# Keep "agibot" in sys.argv for constants detection
if len(sys.argv) == 1:
    sys.argv.append("agibot")

from unifolm_vla.model.framework import build_framework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType

import tensorflow as tf
from qwen_vl_utils import process_vision_info
import numpy as np
import torch
from PIL import Image
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
UNIFOLM_VLA_IMAGE_SIZE = 224


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =======================================
# Helpers
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
    - float32 [0, 1] → uint8 [0, 255]
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # If [C, H, W], convert to [H, W, C]
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # If float in [0,1], convert to uint8 [0, 255]
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    
    return img


def unnormalize_action(normalized_actions: np.ndarray, action_norm_stats: dict):
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.asarray(
            action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)),
            dtype=bool,
        )
        action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
        action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.asarray(
            action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool)),
            dtype=bool,
        )
        action_high = np.asarray(action_norm_stats["q99"], dtype=np.float32)
        action_low = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    normalized_actions = normalized_actions.reshape(-1)
    # Slice stats to match the action dimension
    n_dims = normalized_actions.shape[0]
    action_high = action_high[:n_dims].reshape(-1)
    action_low = action_low[:n_dims].reshape(-1)
    if mask is not None:
        mask = mask[:n_dims].reshape(-1)

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )
    return actions.astype(np.float32)


def normalize_proprio(proprio: np.ndarray, norm_stats: dict):
    proprio = np.asarray(proprio, dtype=np.float32)

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.asarray(
            norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool)),
            dtype=bool,
        )
        proprio_high = np.asarray(norm_stats["max"], dtype=np.float32)
        proprio_low = np.asarray(norm_stats["min"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.asarray(
            norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)),
            dtype=bool,
        )
        proprio_high = np.asarray(norm_stats["q99"], dtype=np.float32)
        proprio_low = np.asarray(norm_stats["q01"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    if proprio.ndim == 1:
        proprio = proprio[None, :]

    # Slice stats to match the proprio dimension
    n_dims = proprio.shape[-1]
    proprio_high = proprio_high[:n_dims]
    proprio_low = proprio_low[:n_dims]
    mask = mask[:n_dims]

    normalized_proprio = np.clip(
        np.where(
            mask[None, :],
            2.0 * (proprio - proprio_low[None, :]) / (proprio_high[None, :] - proprio_low[None, :] + 1e-8) - 1.0,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio.astype(np.float32)


# =======================================
# Load Model
# =======================================

# Trained 32D direct pass-through checkpoint from our latest training (10000 steps)
CHECKPOINT_PATH = "/root/gpufree-data/unifolm-vla/results/Checkpoints/unifolm_vla_agibot_acot/checkpoints/steps_10000_pytorch_model.pt"
VLM_PRETRAINED_PATH = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
DATASET_NORM_KEY = "rlds_dataset"
PORT = 8999
HOST = "0.0.0.0"

logger.info("=" * 60)
logger.info(" UnifoLM-VLA Policy Server (ACoT 32D Direct version) ")
logger.info("=" * 60)
logger.info(f" Checkpoint: {CHECKPOINT_PATH}")
logger.info(f" VLM Pretrained: {VLM_PRETRAINED_PATH}")
logger.info(f" Dataset normalization key: {DATASET_NORM_KEY}")
logger.info(f" Port: {PORT}")
logger.info("")

# Parse config
import argparse
import sys
from omegaconf import OmegaConf

# Accept optional "agibot" argument (for constants detection)
parser = argparse.ArgumentParser()
parser.add_argument("agibot", nargs="?", help="Optional argument for AGIBOT constants detection")
args = parser.parse_args()

# Load config and norm_stats from checkpoint directory
logger.info("Loading config and norm_stats from checkpoint...")
from unifolm_vla.model.framework.share_tools import read_mode_config
model_config, norm_stats = read_mode_config(CHECKPOINT_PATH)

# Convert to OmegaConf
cfg = OmegaConf.create(model_config)

# Update VLM path if needed
if VLM_PRETRAINED_PATH is not None:
    cfg.framework.qwenvl.base_vlm = VLM_PRETRAINED_PATH

logger.info("Loading model...")
from unifolm_vla.model.framework import build_framework
vla = build_framework(cfg)

# Attach norm_stats to model
vla.norm_stats = norm_stats

# Load pre-trained checkpoint
if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):
    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    vla.load_state_dict(checkpoint, strict=True)

logger.info("Converting to bfloat16...")
vla = vla.to(torch.bfloat16)
vla = vla.to(DEVICE).eval()

norm_stats_action = vla.norm_stats[DATASET_NORM_KEY]['action']
norm_stats_proprio = vla.norm_stats[DATASET_NORM_KEY]['proprio']
processor = vla.qwen_vl_interface.processor

# Metadata
metadata = {
    "model_name": "unifolm_vla_acot_direct_32d",
    "action_dim": 32,
    "checkpoint": CHECKPOINT_PATH,
}

logger.info("Model loaded successfully!")
logger.info(f"Metadata: {metadata}")


# =======================================
# Main Inference Handler
# =======================================

async def handler(websocket: ws_server.ServerConnection):
    logger.info("=" * 80)
    logger.info("=" * 80)
    logger.info(f"NEW WEBSOCKET CONNECTION FROM {websocket.remote_address} OPENED!")
    logger.info("=" * 80)
    logger.info("=" * 80)
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()

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

            # Log the full observation structure to see what we're getting (without json.dumps to avoid bytes key issues)
            logger.info(f"  Full observation type: {type(obs)}")
            
            # Helper function to log dict structure
            def log_dict_structure(d, indent=0):
                prefix = "  " * indent
                for k, v in d.items():
                    key_repr = repr(k) if isinstance(k, bytes) else str(k)
                    logger.info(f"{prefix}Key: {key_repr} (type: {type(k)}), Value type: {type(v)}")
                    if isinstance(v, dict) and indent < 3:
                        log_dict_structure(v, indent + 1)
                    elif isinstance(v, list) and len(v) > 0 and indent < 3:
                        logger.info(f"{prefix}  List item 0 type: {type(v[0])}")
            
            if isinstance(obs, dict):
                log_dict_structure(obs)

            # Handle both single observation and batch observations
            if isinstance(obs, list):
                logger.info(f"  Received batched observations: {len(obs)}")
                observations = obs
            else:
                observations = [obs]

            # Collect images
            all_images = []
            for observation in observations:
                found_images = False
                # Check common image keys
                if "observation.images.top_head" in observation:
                    img = process_image_from_obs(observation["observation.images.top_head"])
                    all_images.append(img)
                    found_images = True

                if "image_top_head" in observation:
                    img = process_image_from_obs(observation["image_top_head"])
                    all_images.append(img)
                    found_images = True

                if "full_image" in observation:
                    img = process_image_from_obs(observation["full_image"])
                    all_images.append(img)
                    found_images = True

                if "images" in observation and isinstance(observation["images"], dict):
                    images_dict = observation["images"]
                    if "top_head" in images_dict:
                        # Check what type this is
                        top_head_val = images_dict["top_head"]
                        logger.info(f"  top_head value type: {type(top_head_val)}")
                        # If it's a dict with a 'array' or 'data' key, extract that
                        if isinstance(top_head_val, dict):
                            # Check for msgpack-numpy format (keys as bytes)
                            found = False
                            # Convert bytes keys to str for easier checking
                            top_head_dict_str = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in top_head_val.items()}
                            if "__ndarray__" in top_head_dict_str:
                                # This is msgpack-numpy serialized array
                                data = top_head_dict_str["data"]
                                dtype = top_head_dict_str["dtype"]
                                shape = top_head_dict_str["shape"]
                                # Convert dtype from bytes to str if needed
                                if isinstance(dtype, bytes):
                                    dtype = dtype.decode('utf-8')
                                # Reconstruct numpy array
                                img_arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                                img = process_image_from_obs(img_arr)
                                found = True
                            elif "array" in top_head_dict_str:
                                img = process_image_from_obs(top_head_dict_str["array"])
                                found = True
                            elif "data" in top_head_dict_str:
                                img = process_image_from_obs(top_head_dict_str["data"])
                                found = True
                            elif "image" in top_head_dict_str:
                                img = process_image_from_obs(top_head_dict_str["image"])
                                found = True
                            else:
                                # Try to find any key that looks like image data
                                for k, v in top_head_val.items():
                                    if isinstance(v, (np.ndarray, torch.Tensor)):
                                        img = process_image_from_obs(v)
                                        found = True
                                        break
                            if not found:
                                raise ValueError(f"Couldn't find image data in top_head dict: {top_head_val.keys()}")
                        else:
                            img = process_image_from_obs(top_head_val)
                        all_images.append(img)
                    # Check for wrist images
                    for img_k, img_v in images_dict.items():
                        if isinstance(img_k, bytes):
                            img_k_str = img_k.decode('utf-8')
                        else:
                            img_k_str = img_k
                        if "wrist" in img_k_str.lower() or "hand" in img_k_str.lower():
                            if isinstance(img_v, dict):
                                # Check for msgpack-numpy format
                                img_v_dict_str = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in img_v.items()}
                                found = False
                                if "__ndarray__" in img_v_dict_str:
                                    data = img_v_dict_str["data"]
                                    dtype = img_v_dict_str["dtype"]
                                    shape = img_v_dict_str["shape"]
                                    if isinstance(dtype, bytes):
                                        dtype = dtype.decode('utf-8')
                                    img_arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                                    img = process_image_from_obs(img_arr)
                                    found = True
                                elif "array" in img_v_dict_str:
                                    img = process_image_from_obs(img_v_dict_str["array"])
                                    found = True
                                elif "data" in img_v_dict_str:
                                    img = process_image_from_obs(img_v_dict_str["data"])
                                    found = True
                                elif "image" in img_v_dict_str:
                                    img = process_image_from_obs(img_v_dict_str["image"])
                                    found = True
                                else:
                                    for k, v in img_v.items():
                                        if isinstance(v, (np.ndarray, torch.Tensor)):
                                            img = process_image_from_obs(v)
                                            found = True
                                            break
                                if not found:
                                    continue
                            else:
                                img = process_image_from_obs(img_v)
                            all_images.append(img)

            instruction = observations[0].get("instruction", "")
            if not instruction and "prompt" in observations[0]:
                instruction = observations[0]["prompt"]

            logger.info(f"  Found {len(all_images)} images")
            logger.info(f"  Instruction: {instruction}")

            # Make sure we have at least one image
            if not all_images:
                raise ValueError(f"No images found in observation! keys={list(observations[0].keys())}")

            # =======================================
            # Process images
            # =======================================
            processed_images = []
            for image in all_images:
                check_image_format(image)
                if image.shape != (UNIFOLM_VLA_IMAGE_SIZE, UNIFOLM_VLA_IMAGE_SIZE, 3):
                    image = resize_image_for_policy(image, UNIFOLM_VLA_IMAGE_SIZE)
                pil_image = Image.fromarray(image).convert("RGB")
                processed_images.append(pil_image)

            # =======================================
            # Build prompt
            # =======================================
            lang = instruction.lower()
            text = f'The task is "{lang}".'
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
            # Process inputs
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
            # Process proprioception
            # =======================================
            proprios = []
            for observation in observations:
                if "observation.state" in observation:
                    state = observation["observation.state"]
                elif "state" in observation:
                    state = observation["state"]
                else:
                    # If no state found, use zeros - 32D
                    state = np.zeros(32, dtype=np.float32)
                    logger.warning("No state found in observation! Using zeros (32D)")

                # Check if state is a msgpack-numpy dict
                if isinstance(state, dict):
                    state_dict_str = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in state.items()}
                    if "__ndarray__" in state_dict_str:
                        data = state_dict_str["data"]
                        dtype = state_dict_str["dtype"]
                        shape = state_dict_str["shape"]
                        if isinstance(dtype, bytes):
                            dtype = dtype.decode('utf-8')
                        state = np.frombuffer(data, dtype=dtype).reshape(shape)

                # Already converted to 32D by adapter during training - just ensure it's 32D
                state = np.asarray(state, dtype=np.float32).reshape(-1)
                if state.shape[0] > 32:
                    state = state[:32]
                elif state.shape[0] < 32:
                    tmp = np.zeros(32, dtype=np.float32)
                    tmp[:state.shape[0]] = state
                    state = tmp

                proprios.append(state)

            batch_input["state"] = torch.from_numpy(
                normalize_proprio(np.stack(proprios, axis=0), norm_stats_proprio)
            ).unsqueeze(0).to(DEVICE)

            # =======================================
            # Move everything to device
            # =======================================
            for key, value in batch_input.items():
                if value is not None:
                    batch_input[key] = value.to(DEVICE)

            # =======================================
            # Predict action
            # =======================================
            infer_time = time.monotonic()
            with torch.inference_mode():
                action = vla.predict_action(qwen_inputs=batch_input)
            infer_time = time.monotonic() - infer_time

            action_32 = unnormalize_action(action["normalized_actions"][0], norm_stats_action)
            assert action_32.shape[0] == 32, f"Expected 32D action, got {action_32.shape}"

            # Genie_sim expects (1, 32)
            result = {
                "actions": action_32[None, :],
                "server_timing": {
                    "infer_ms": infer_time * 1000,
                },
            }
            if prev_total_time is not None:
                result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

            logger.info(f"  Generated action shape: {result['actions'].shape}")
            logger.info("=" * 80)

            await websocket.send(packer.pack(result))
            prev_total_time = time.monotonic() - start_time

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
            break
        except Exception:
            err = traceback.format_exc()
            logger.error(f"Error: {err}")
            try:
                await websocket.send(err)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
            except Exception:
                pass
            break


async def health_check(connection, request):
    """Health check endpoint for HTTP."""
    if request.path == "/healthz":
        return connection.respond(200, "OK\n")
    return None


async def main():
    logger.info(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    logger.info(f"Health check: http://{HOST}:{PORT}/healthz")
    server = await ws_server.serve(
        handler,
        HOST,
        PORT,
        compression=None,
        max_size=None,
        process_request=health_check,
    )
    await server.wait_closed()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
