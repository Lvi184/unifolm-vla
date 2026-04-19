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
 Output keys match: input_ids, pixel_values, actions, proprio
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

 # Remove batch dimension added by processor
 out = {
 "input_ids": proc["input_ids"][0],
 "pixel_values": proc["pixel_values"][0],
 "actions": torch.tensor(sample["action"], dtype=torch.float32), # collate_fn expects "actions" (plural)
 }

 if "image_grid_thw" in proc:
 out["image_grid_thw"] = proc["image_grid_thw"][0]

 if self.use_proprio:
 out["proprio"] = torch.tensor(obs["proprio"], dtype=torch.float32)

 return out
