import numpy as np

# ACoT Go2/G2A official slicing rules:
# state 159 -> 21: list(range(30, 44)) + [0, 1] + list(range(75, 80))
# action 40 -> 21: [16:30] + [0:2] + [33:38]
STATE_IDX_159_TO_21: list[int] = list(range(30, 44)) + [0, 1] + list(range(75, 80))
ACTION_SLICES_40_TO_21: tuple[tuple[int, int], ...] = (
    (16, 30),  # 14 joint positions
    (0, 2),    # 2 left/right effector
    (33, 38),  # 5 waist positions
)

def state159_to_21(state159: np.ndarray) -> np.ndarray:
    """Project full 159D G2A state to 21D control state following ACoT rules."""
    state159 = np.asarray(state159, dtype=np.float32)
    assert state159.shape[-1] == 159, f"Expected 159D state, got {state159.shape[-1]}D"
    return np.ascontiguousarray(state159[STATE_IDX_159_TO_21], dtype=np.float32)

def action40_to_21(action40: np.ndarray) -> np.ndarray:
    """Project full 40D G2A action to 21D control action following ACoT rules."""
    action40 = np.asarray(action40, dtype=np.float32)
    assert action40.shape[-1] == 40, f"Expected 40D action, got {action40.shape[-1]}D"
    chunks = [action40[start:end] for start, end in ACTION_SLICES_40_TO_21]
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

def batch_state159_to_21(states159: np.ndarray) -> np.ndarray:
    """Batch version for [T, 159] -> [T, 21]"""
    states159 = np.asarray(states159, dtype=np.float32)
    assert states159.ndim == 2 and states159.shape[1] == 159
    return np.ascontiguousarray(states159[:, STATE_IDX_159_TO_21], dtype=np.float32)

def batch_action40_to_21(actions40: np.ndarray) -> np.ndarray:
    """Batch version for [T, 40] -> [T, 21]"""
    actions40 = np.asarray(actions40, dtype=np.float32)
    assert actions40.ndim == 2 and actions40.shape[1] == 40
    chunks = [
        actions40[:, start:end] for start, end in ACTION_SLICES_40_TO_21
    ]
    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)

def get_mapping_info() -> dict:
    """Return human-readable mapping info for debugging."""
    state_info = []
    for i, src in enumerate(STATE_IDX_159_TO_21):
        state_info.append((i, src))
    action_info = []
    idx = 0
    for start, end in ACTION_SLICES_40_TO_21:
        for src in range(start, end):
            action_info.append((idx, src))
            idx += 1
    return {
        "state_mapping": state_info,
        "action_mapping": action_info,
        "state_dim": 21,
        "action_dim": 21,
    }

if __name__ == "__main__":
    info = get_mapping_info()
    print("STATE 159 -> 21 mapping:")
    for ctrl_i, src_i in info["state_mapping"]:
        print(f"  ctrl[{ctrl_i:02d}] <- full[{src_i:03d}]")
    print("\nACTION 40 -> 21 mapping:")
    for ctrl_i, src_i in info["action_mapping"]:
        print(f"  ctrl[{ctrl_i:02d}] <- full[{src_i:02d}]")
