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


def _state_to_32(state: np.ndarray) -> np.ndarray:
    """Convert any input state to 32D, matching genie_sim competition interface.
    
    - If input is already 32D, keep it as-is
    - If input > 32D, take first 32
    - If input < 32D, pad with zeros
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == 32:
        return state
    out = np.zeros(32, dtype=np.float32)
    copy_dim = min(state.shape[0], 32)
    out[:copy_dim] = state[:copy_dim]
    return out


def _action_to_32(action: np.ndarray) -> np.ndarray:
    """Convert any input action to 32D, matching genie_sim competition interface.
    
    - If input is already 32D, keep it as-is
    - If input >= 40D, take first 32
    - If input < 32D, pad with zeros
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 32:
        return action
    out = np.zeros(32, dtype=np.float32)
    copy_dim = min(action.shape[0], 32)
    out[:copy_dim] = action[:copy_dim]
    return out


@dataclass
class ACOTAdapterConfig:
    use_left_wrist: bool = True
    use_right_wrist: bool = True
    project_state_to_21: bool = False  # Disabled for 32D direct pass-through
    project_action_to_21: bool = False  # Disabled for 32D direct pass-through


class ACOTToUnifoLMAdapter:
    def __init__(self, cfg: ACOTAdapterConfig):
        self.cfg = cfg

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        obs = sample["observation"]
        imgs = obs["images"]

        state = obs["state"]
        action = sample["action"]

        # Always convert to 32D first
        state = _state_to_32(state)
        action = _action_to_32(action)

        # Optional projection for backwards compatibility (disabled by default for 32D direct)
        if self.cfg.project_state_to_21:
            from .acot_adapter import state159_to_21
            state = state159_to_21(state)

        if self.cfg.project_action_to_21:
            from .acot_adapter import action40_to_21
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
