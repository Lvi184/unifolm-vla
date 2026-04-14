"""
Simple test script to verify UnifoLM-VLA model inference works.
"""
import sys
import logging

# Force G2A constants
sys.argv.append("agibot")

import numpy as np
import torch
from PIL import Image
import time

# Import UnifoLM-VLA
from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType

import tensorflow as tf
from qwen_vl_utils import process_vision_info

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
unifolm_vla_IMAGE_SIZE = 224


# Helper functions
def unnormalize_action(normalized_actions: np.ndarray, action_norm_stats: dict):
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


def normalize_proprio(proprio: np.ndarray, norm_stats: dict):
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


def check_image_format(image):
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        f"Incorrect image format! Need (H, W, 3) uint8, got shape={image.shape}, dtype={image.dtype}"
    )


def resize_image_for_policy(img, resize_size):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configuration
    CHECKPOINT_PATH = "./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt"
    VLM_PRETRAINED_PATH = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
    UNNORM_KEY = "rlds_dataset"
    
    logger.info("=" * 60)
    logger.info(" UnifoLM-VLA Model Inference Test")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"VLM Pretrained: {VLM_PRETRAINED_PATH}")
    logger.info(f"Dataset key: {UNNORM_KEY}")
    logger.info("")
    
    # Load model
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
    
    logger.info("Model loaded successfully!")
    
    # Create dummy inputs
    logger.info("")
    logger.info("Creating dummy inputs...")
    
    # Dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    check_image_format(dummy_image)
    
    # Process image
    processed_images = []
    pil_image = Image.fromarray(dummy_image).convert("RGB")
    processed_images.append(pil_image)
    
    # Dummy instruction
    instruction = "pick up the object"
    
    # Build prompt
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
    
    # Process inputs
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
    
    # Dummy state (159D)
    dummy_state = np.random.rand(159).astype(np.float32)
    batch_input["state"] = torch.from_numpy(
        normalize_proprio(np.stack([dummy_state], axis=0), norm_stats_proprio)
    ).to(DEVICE)
    
    # Move everything to device
    batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
    batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
    batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
    batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)
    
    logger.info("Running inference...")
    t1 = time.time()
    
    # Predict action
    action = vla.predict_action(qwen_inputs=batch_input)
    action = unnormalize_action(action['normalized_actions'][0], norm_stats_action)
    
    inference_time = time.time() - t1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Inference SUCCESS!")
    logger.info("=" * 60)
    logger.info(f"Inference time: {inference_time:.3f}s")
    logger.info(f"Action shape: {action.shape}")
    logger.info(f"Action sample (first 10): {action[:10]}")
    logger.info("")
    logger.info("Model works perfectly!")


if __name__ == "__main__":
    main()
