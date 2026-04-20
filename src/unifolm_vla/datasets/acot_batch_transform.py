from __future__ import annotations

from typing import Dict, Any
import numpy as np
import torch
from PIL import Image


def to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return Image.fromarray(img)


class ACOTBatchTransform:
    """
    Transform that converts adapted sample dict
    into the format expected by UnifoLM training collate_fn.
    Output keys match what collate_fn expects: input_ids, pixel_values, actions, proprio
    """

    def __init__(self, processor, use_wrist_image=True, use_proprio=True):
        self.processor = processor
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        obs = sample["observation"]
        instruction = sample["task"]["language_instruction"]

        # Primary image always exists
        image_primary = to_pil(obs["image_primary"])

        # Build Qwen-VL chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process with processor
        proc = self.processor(
            text=[text],
            images=[image_primary],
            return_tensors="pt",
        )

        # proc already has shape [1, ...] from processor(text=[text], ...) so we keep first dimension
        # For actions: ensure it's [1, 21] for stacking
        action_tensor = torch.tensor(sample["action"], dtype=torch.float32)
        # ensure it's 2D [1, 21] regardless of input shape
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim != 2:
            action_tensor = action_tensor.reshape(1, -1)
        
        out = {
            "input_ids": proc["input_ids"][0],  # input_ids: [1, seq_len] -> [seq_len]
            "pixel_values": proc["pixel_values"],  # keep [1, ...] for cat after collate
            "actions": action_tensor,  # shape [1, 21] -> collate.stack -> [batch, 1, 21] -> np.squeeze -> [batch, 21] -> DiT adds [batch, 1, 21]
        }

        if "image_grid_thw" in proc:
            # processor returns [1, 3] for one image -> keep it -> collate.cat -> [batch, 3] correct
            out["image_grid_thw"] = proc["image_grid_thw"]

        if self.use_proprio:
            proprio_tensor = torch.tensor(obs["proprio"], dtype=torch.float32)
            # ensure it's 1D [21] regardless of input shape
            if proprio_tensor.ndim != 1:
                proprio_tensor = proprio_tensor.reshape(-1)
            out["proprio"] = proprio_tensor  # shape [21] -> collate.stack -> [batch, 21] correct for DiT state_encoder input

        return out
