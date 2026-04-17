#!/usr/bin/env python3
"""
Compute normalization statistics (q01, q99) for agibot_g2a_21 dataset.
Saves results to ${UNIFOLM_HDF5_ROOT}/agibot_g2a_21_norm_stats.json
Results match what UnifoLM expects for BOUNDS_Q99 normalization.
"""

import os
import json
import glob
from pathlib import Path
import h5py
import numpy as np

def main():
    hdf5_root = Path(os.environ.get(
        "UNIFOLM_HDF5_ROOT",
        "/root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5",
    ))

    all_actions = []
    all_proprio = []

    # Iterate all tasks and all episodes
    for task_dir in sorted(hdf5_root.iterdir()):
        if not task_dir.is_dir():
            continue
        print(f"Processing task: {task_dir.name}")

        # Find all HDF5 files in this task
        for h5_path in sorted(task_dir.glob("*.hdf5")):
            try:
                with h5py.File(h5_path, "r") as f:
                    actions = f["action"][:]             # [T, 21]
                    proprio = f["observations"]["qpos"][:]  # [T, 21]

                if actions.ndim == 2 and actions.shape[1] == 21:
                    all_actions.append(actions)
                if proprio.ndim == 2 and proprio.shape[1] == 21:
                    all_proprio.append(proprio)
            except Exception as e:
                print(f"  ERROR reading {h5_path}: {e}")
                continue

    # Concatenate all data
    actions = np.concatenate(all_actions, axis=0)
    proprio = np.concatenate(all_proprio, axis=0)

    # Compute statistics
    stats = {
        "agibot_g2a_21": {
            "action": {
                "q01": np.quantile(actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions, 0.99, axis=0).tolist(),
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
                "mask": np.ones(actions.shape[1], dtype=bool).tolist(),
            },
            "state": {
                "q01": np.quantile(proprio, 0.01, axis=0).tolist(),
                "q99": np.quantile(proprio, 0.99, axis=0).tolist(),
                "min": proprio.min(axis=0).tolist(),
                "max": proprio.max(axis=0).tolist(),
                "mask": np.ones(proprio.shape[1], dtype=bool).tolist(),
            },
        },
    }

    # Save
    out_path = hdf5_root / "agibot_g2a_21_norm_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Saved to: {out_path}")
    print(f"   Total action frames: {actions.shape[0]}")
    print(f"   Total proprio frames: {proprio.shape[0]}")
    print(f"   Action dimension: {actions.shape[1]}")
    print(f"   Proprio dimension: {proprio.shape[1]}")

if __name__ == "__main__":
    main()
