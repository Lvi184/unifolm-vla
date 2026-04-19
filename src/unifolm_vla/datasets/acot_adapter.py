from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


STATE_IDX_159_TO_21 = list(range(30, 44)) + [0, 1] + list(range(75, 80))


def state159_to_21(state159: np.ndarray) -> np.ndarray:
    state159 = np.asarray(state159, dtype=np.float32)
    if state159.shape[-1] != 159:
        raise ValueError(f"Expected 159D state, got {state159.shape}")
    return state159[STATE_IDX_159_TO_21].astype(np.float32)


def action40_to_21(action40: np.ndarray) -> np.ndarray:
    action40 = np.asarray(action40, dtype=np.float32)
    if action40.shape[-1] != 40:
        raise ValueError(f"Expected 40D action, got {action40.shape}")
    return np.concatenate(
        [
            action40[16:30],  # 14
            action40[0:2],  # 2
            action40[33:38],  # 5
        ],
        axis=0,
    ).astype(np.float32)


@dataclass
class ACOTAdapterConfig:
    use_left_wrist: bool = True
    use_right_wrist: bool = True
    project_state_to_21: bool = True
    project_action_to_21: bool = True


class ACOTToUnifoLMAdapter:
    def __init__(self, cfg: ACOTAdapterConfig):
        self.cfg = cfg

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        obs = sample["observation"]
        imgs = obs["images"]

        state = obs["state"]
        action = sample["action"]

        if self.cfg.project_state_to_21:
            state = state159_to_21(state)

        if self.cfg.project_action_to_21:
            action = action40_to_21(action)

        out_obs: Dict[str, Any] = {
            "image_primary": imgs["top_head"],
            "proprio": state,
        }

        if self.cfg.use_left_wrist and imgs.get("hand_left") is not None:
            out_obs["image_left_wrist"] = imgs["hand_left"]

        if self.cfg.use_right_wrist and imgs.get("hand_right") is not None:
            out_obs["image_right_wrist"] = imgs["hand_right"]
        # 兼容一些老逻辑只认 image_wrist
        out_obs["image_wrist"] = imgs["hand_right"]

        return {
            "observation": out_obs,
            "action": action,
            "task": {
                "language_instruction": sample["instruction"],
            },
            "episode_id": sample["episode_id"],
            "frame_index": sample["frame_index"],
        }
