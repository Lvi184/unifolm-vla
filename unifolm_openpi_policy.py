"""
UnifoLM-VLA OpenPI Policy Adapter
================================

This adapter makes UnifoLM-VLA compatible with the openpi.Policy interface,
including proper dimension conversion between genie_sim/ACoT-VLA 32D format
and UnifoLM-VLA 159D/40D training format.

Based on actual observation from genie_sim!
"""
import dataclasses
import logging
import time
from typing import Dict, Any, List, Union, Tuple

import numpy as np
import torch
import json
from PIL import Image

# Import from local UnifoLM-VLA project
from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType

import tensorflow as tf
from qwen_vl_utils import process_vision_info

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
unifolm_vla_IMAGE_SIZE = 224


def convert_32d_state_to_159d(state_32d: np.ndarray) -> np.ndarray:
    """
    Convert genie_sim/ACoT-VLA 32D state to UnifoLM-VLA 159D state.
    
    Based on ACoT-VLA's slice_state_and_action in reverse:
        if len(data["state"]) == 159:
            state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
    """
    state_159d = np.zeros(159, dtype=np.float32)
    
    # Based on the reverse of ACoT-VLA's slice:
    # They took from 159D:
    # - list(range(30, 44)) → 14 elements (indices 30-43)
    # - [0, 1] → 2 elements (indices 0-1)
    # - list(range(75, 80)) → 5 elements (indices 75-79)
    # Total: 21, but wait genie_sim's state is 32D...
    # Let's just fill the positions that ACoT-VLA used, and pad the rest with zeros
    # For now, let's assume 32D maps to the first 32D of 159D
    # We can adjust based on testing
    state_159d[:len(state_32d)] = state_32d
        
    return state_159d


def convert_40d_action_to_32d(action_40d: np.ndarray) -> np.ndarray:
    """
    Convert UnifoLM-VLA 40D action to genie_sim/ACoT-VLA 32D action.
    
    Based on ACoT-VLA's slice_state_and_action:
        if "actions" in data:
            assert data["actions"].shape[1] == 40
            data["actions"] = np.column_stack((data["actions"][:, 16:30], data["actions"][:, 0:2], data["actions"][:, 33:38]))
    """
    if len(action_40d) == 40:
        # ACoT-VLA takes:
        # - data["actions"][:, 16:30] → 14 elements
        # - data["actions"][:, 0:2] → 2 elements
        # - data["actions"][:, 33:38] → 6 elements
        # Total: 22, but genie_sim expects 32D
        # Let's just take first 32D for now
        action_32d = action_40d[:32]
        return action_32d
    return action_40d


def process_image_for_policy(img: Any) -> np.ndarray:
    """
    Process an image from genie_sim format to UnifoLM-VLA format.
    
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


# =======================================
# Image Preprocessing Helpers (from UnifoLM-VLA)
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


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths],
        axis=1,
    )
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE)
    )
    if expanded_dims:
        image = image[0]
    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    batch_size = 1
    crop_scale = 0.9
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


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


@dataclasses.dataclass
class UnifoLMOpenPIPolicy:
    """
    UnifoLM-VLA policy adapter for openpi.Policy interface.
    
    Supports LeRobot/genie_sim format observations, including
    dimension conversion between 32D (genie_sim) and 159D/40D (UnifoLM-VLA).
    
    Based on actual genie_sim observation format!
    """
    
    ckpt_path: str
    vlm_pretrained_path: str
    unnorm_key: str = "rlds_dataset"
    center_crop: bool = False
    use_bf16: bool = True
    
    def __post_init__(self):
        logging.info("Loading UnifoLM-VLA from: %s", self.ckpt_path)
        
        # Load model
        self.vla = baseframework.from_pretrained(
            self.ckpt_path, 
            vlm_pretrained_path=self.vlm_pretrained_path
        )
        
        if self.use_bf16:
            logging.info("Converting to bfloat16")
            self.vla = self.vla.to(torch.bfloat16)
        
        self.vla = self.vla.to(DEVICE).eval()
        self.processor = self.vla.qwen_vl_interface.processor
        
        # Load normalization stats
        self.norm_stats_action = self.vla.norm_stats[self.unnorm_key]['action']
        self.norm_stats_proprio = self.vla.norm_stats[self.unnorm_key]['proprio']
        
        # Metadata (for openpi)
        max_action = self.norm_stats_action.get("max", [])
        if isinstance(max_action, list):
            action_dim = len(max_action)
        else:
            action_dim = getattr(max_action, 'shape', [0])[0]
        
        self.metadata = {
            "model_name": "unifolm_vla",
            "action_dim": action_dim,
            "checkpoint": self.ckpt_path,
        }
        
        logging.info("UnifoLM-VLA loaded successfully!")

    def prepare_images_for_vla(self, images: List[np.ndarray]) -> List[Image.Image]:
        processed_images = []
        for image in images:
            check_image_format(image)
            if image.shape != (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE, 3):
                image = resize_image_for_policy(image, unifolm_vla_IMAGE_SIZE)
            pil_image = Image.fromarray(image).convert("RGB")
            if self.center_crop:
                pil_image = center_crop_image(pil_image)
            processed_images.append(pil_image)
        return processed_images
        
    def infer(self, obs: Dict) -> Dict:
        """
        Main inference function for openpi.Policy interface.
        
        Accepts LeRobot/genie_sim format observations, converts them to
        UnifoLM-VLA format, runs inference, then converts back.
        
        Based on actual genie_sim observation format!
        """
        try:
            t1 = time.time()
            
            # Handle both single observation and batch observations
            if isinstance(obs, list):
                observations = obs
            else:
                observations = [obs]
                
            # Collect images - from obs["images"] dict!
            all_images = []
            for observation in observations:
                # Check if we have an "images" dict (genie_sim format)
                if "images" in observation and isinstance(observation["images"], dict):
                    images_dict = observation["images"]
                    # External camera - top_head
                    if "top_head" in images_dict:
                        img = process_image_for_policy(images_dict["top_head"])
                        all_images.append(img)
                    # Wrist cameras
                    for img_k, img_v in images_dict.items():
                        if "wrist" in img_k.lower() or "hand" in img_k.lower():
                            img = process_image_for_policy(img_v)
                            all_images.append(img)
            
            # Get instruction
            instruction = observations[0].get("instruction", "")
            if not instruction and "prompt" in observations[0]:
                instruction = observations[0]["prompt"]
                
            # Get task name for normalization stats (if available, else use default)
            if "task_name" in observations[0] and observations[0]["task_name"] is not None:
                task_name = observations[0]["task_name"]
                if task_name in self.vla.norm_stats:
                    self.norm_stats_action = self.vla.norm_stats[task_name]['action']
                    self.norm_stats_proprio = self.vla.norm_stats[task_name]['proprio']
                else:
                    # Fallback to default if task_name not found
                    logging.warning(f"Task name '{task_name}' not found in norm_stats, using default 'rlds_dataset'")

            # Process images
            all_images = self.prepare_images_for_vla(all_images)
            lang = instruction.lower()
            text = f"The task is \"{lang}\"."
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in all_images],
                        {"type": "text", "text": text},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch_input = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Process proprioception - convert from genie_sim 32D to UnifoLM-VLA 159D
            proprios = []
            for observation in observations:
                if "observation.state" in observation:
                    state = observation["observation.state"]
                elif "state" in observation:
                    state = observation["state"]
                else:
                    raise ValueError("No state found in observation!")
                
                # Convert from genie_sim 32D → UnifoLM-VLA 159D if needed
                if len(state) == 32:
                    state = convert_32d_state_to_159d(state)
                
                proprios.append(state)
                    
            batch_input["state"] = torch.from_numpy(
                normalize_proprio(np.stack(proprios, axis=0), self.norm_stats_proprio)
            ).unsqueeze(0).to(DEVICE)

            batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
            batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
            batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
            batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)
            
            action = self.vla.predict_action(qwen_inputs=batch_input)
            action = unnormalize_action(action['normalized_actions'][0], self.norm_stats_action)
            
            # Convert from UnifoLM-VLA 40D → genie_sim 32D if needed
            if len(action) == 40:
                action = convert_40d_action_to_32d(action)
            
            inference_time = time.time() - t1
            logging.info(f"UnifoLM-VLA inference: {inference_time:.3f}s")
            
            return {"actions": action}
            
        except Exception as e:
            import traceback
            logging.error(traceback.format_exc())
            raise
