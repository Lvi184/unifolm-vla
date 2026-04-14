"""
Debug script to print the shapes of the tensors in DiT_ActionHeader.py
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
    logger.info(" UnifoLM-VLA Tensor Shape Debug")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
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
    
    # Dummy state (159D) - NOTE: this is what causes the issue!
    dummy_state = np.random.rand(159).astype(np.float32)
    batch_input["state"] = torch.from_numpy(
        normalize_proprio(np.stack([dummy_state], axis=0), norm_stats_proprio)
    ).unsqueeze(0).to(DEVICE)
    
    # Move everything to device
    batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
    batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
    batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
    batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)
    
    logger.info("Running forward pass to debug shapes...")
    
    # Get vl_embs first
    with torch.no_grad():
        vl_embs = vla.vlm.get_vl_embeddings(batch_input)
    
    logger.info(f"vl_embs shape: {vl_embs.shape}")
    
    # Let's look into the action_model
    action_model = vla.action_model
    
    # Let's manually call the predict_action function
    # First, create initial actions
    batch_size = vl_embs.shape[0]
    device = vl_embs.device
    
    actions = torch.zeros(
        size=(batch_size, action_model.action_horizon, action_model.action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )
    
    logger.info(f"Initial actions shape: {actions.shape}")
    
    # Get state
    state = batch_input["state"]
    
    # First pass to debug
    state_features = action_model.state_encoder(state).unsqueeze(1) if state is not None else None
    logger.info(f"state_features shape: {state_features.shape if state_features is not None else 'None'}")
    
    # Run one denoising step to see
    num_steps = 1
    dt = 1.0 / num_steps
    
    for t in range(num_steps):
        logger.info(f"\n=== Step {t} ===")
        t_cont = t / float(num_steps)
        t_discretized = int(t_cont * action_model.num_timestep_buckets)
        
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        
        action_features = action_model.action_encoder(actions, timesteps_tensor)
        logger.info(f"action_features shape: {action_features.shape}")
        
        future_tokens = action_model.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        logger.info(f"future_tokens shape: {future_tokens.shape}")
        
        logger.info("")
        logger.info(f"Attempting to concatenate:")
        if state_features is not None:
            logger.info(f"  state_features: {state_features.shape}")
        logger.info(f"  future_tokens: {future_tokens.shape}")
        logger.info(f"  action_features: {action_features.shape}")
        logger.info(f"  dim=1")
        
        # Try to concatenate and see the error
        try:
            if state_features is not None:
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            else:
                sa_embs = torch.cat((future_tokens, action_features), dim=1)
            logger.info(f"SUCCESS! sa_embs shape: {sa_embs.shape}")
        except Exception as e:
            logger.error(f"ERROR concatenating: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Let's try to debug the issue - maybe state_features is 4D?
            logger.info("")
            logger.info("Let's check state_features again:")
            if state_features is not None:
                logger.info(f"  state_features ndim: {state_features.ndim}")
                logger.info(f"  state_features shape: {state_features.shape}")
                
                # Let's squeeze any extra dimensions!
                state_features = state_features.squeeze(1)
                logger.info(f"  After squeeze: {state_features.shape}")
                
                # Try again
                try:
                    sa_embs = torch.cat((state_features.unsqueeze(1), future_tokens, action_features), dim=1)
                    logger.info(f"SUCCESS after squeeze! sa_embs shape: {sa_embs.shape}")
                except Exception as e2:
                    logger.error(f"Still error: {e2}")
    
    logger.info("\nDebug complete!")


if __name__ == "__main__":
    main()
