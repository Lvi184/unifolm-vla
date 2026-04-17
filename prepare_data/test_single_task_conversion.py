#!/usr/bin/env python3
"""
Test script to convert a single task and check if TFDS build works.
Usage:
  python test_single_task_conversion.py \
    --task_root /root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth/open_door \
    --hdf5_out /tmp/test_unifolm_hdf5 \
    --num_episodes 5
"""

import argparse
import os
from pathlib import Path
import numpy as np
from adapters.g2a_space_adapter import (
    state159_to_21,
    action40_to_21,
)
from adapters.agibotworld_meta import (
    load_task_meta,
    get_instruction_for_frame,
)
from adapters.agibotworld_reader import (
    load_episode,
)


def convert_single_task(
    task_root: Path,
    hdf5_out_dir: Path,
    num_episodes: int = 5,
):
    """Convert first N episodes of a single task to HDF5, for quick testing."""
    info = load_task_meta(task_root)
    total_episodes = info.get("total_episodes", 0)
    num_episodes = min(num_episodes, total_episodes)
    print(f"=== Testing conversion for task: {task_root.name} ===")
    print(f"Total episodes in task: {total_episodes}, testing first {num_episodes}")
    print(f"Output HDF5 directory: {hdf5_out_dir}")
    print()

    hdf5_out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0

    for episode_id in range(num_episodes):
        print(f"\n--- Episode {episode_id + 1}/{num_episodes} ---")
        try:
            steps = load_episode(task_root, info, episode_id)
            print(f"  Loaded {len(steps)} steps")

            # Validate projection
            for step_idx, step in enumerate(steps):
                s = step["state159"]
                a = step["action40"]
                assert s.shape == (159,), f"state159 shape {s.shape} != (159,)"
                assert a.shape == (40,), f"action40 shape {a.shape} != (40,)"
                s21 = state159_to_21(s)
                a21 = action40_to_21(a)
                assert s21.shape == (21,), f"state21 shape {s21.shape} != (21,)"
                assert a21.shape == (21,), f"action21 shape {a21.shape} != (21,)"
                assert np.all(np.isfinite(s21)), f"state21 has NaN/inf at step {step_idx}"
                assert np.all(np.isfinite(a21)), f"action21 has NaN/inf at step {step_idx}"

            # Write HDF5
            import h5py
            episode_len = len(steps)
            images_buffers = {
                "primary": [],
                "left_wrist": [],
                "right_wrist": [],
            }
            qpos = []
            qvel = []
            actions = []
            instructions = []

            for step in steps:
                step_images = step["images"]
                assert "top_head" in step_images, "missing top_head"
                assert "hand_left" in step_images, "missing hand_left"
                assert "hand_right" in step_images, "missing hand_right"
                
                images_buffers["primary"].append(step_images["top_head"])
                images_buffers["left_wrist"].append(step_images["hand_left"])
                images_buffers["right_wrist"].append(step_images["hand_right"])

                state21 = state159_to_21(step["state159"])
                action21 = action40_to_21(step["action40"])
                instruction = get_instruction_for_frame(info, episode_id, step["frame_index"])

                qpos.append(state21)
                qvel.append(np.zeros_like(state21, dtype=np.float32))
                actions.append(action21)
                instructions.append(instruction)

            # Stack everything
            qpos = np.stack(qpos, axis=0).astype(np.float32)
            qvel = np.stack(qvel, axis=0).astype(np.float32)
            actions = np.stack(actions, axis=0).astype(np.float32)
            images_stacked = {}
            for name, buffer in images_buffers.items():
                assert len(buffer) == episode_len, \
                    f"Camera {name} has {len(buffer)} frames != {episode_len}"
                images_stacked[name] = np.stack(buffer, axis=0).astype(np.uint8)

            # Write HDF5
            out_path = hdf5_out_dir / f"episode_{episode_id:06d}.hdf5"
            with h5py.File(out_path, "w") as f:
                obs = f.create_group("observations")
                imgs = obs.create_group("images")
                for name, data in images_stacked.items():
                    imgs.create_dataset(name, data=data, compression="gzip", compression_opts=9)
                obs.create_dataset("qpos", data=qpos, compression="gzip")
                obs.create_dataset("qvel", data=qvel, compression="gzip")
                f.create_dataset("action", data=actions, compression="gzip")
                f.create_dataset("is_edited", data=np.array([0], dtype=np.uint8))
                main_instruction = instructions[0] if instructions else ""
                f.create_dataset("language_raw", data=main_instruction)
                sub = f.create_dataset(
                    "substep_reasonings",
                    shape=(episode_len,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="gzip",
                )
                sub[:] = instructions

            print(f"  ✓ OK: wrote to {out_path}")
            print(f"    shapes: qpos={qpos.shape}, action={actions.shape}, "
                  f"primary={images_stacked['primary'].shape}")
            success += 1

        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            failed += 1

    print(f"\n=== Conversion test finished ===")
    print(f"Success: {success}, Failed: {failed}")
    if failed == 0:
        print("\n✅ All test episodes converted successfully!")
        print("Next step:")
        print(f"  export UNIFOLM_HDF5_ROOT={hdf5_out_dir.parent}")
        print("  cd prepare_data/hdf5_to_rlds/rlds_dataset")
        print("  tfds build --data_dir /tmp/test_unifolm_rlds --max_examples_per_split 10")
    else:
        print("\n❌ Some episodes failed, fix the errors above before proceeding.")

    return failed == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_root",
        type=str,
        required=True,
        help="Path to task directory (e.g. .../agibot_data_without_depth/open_door)",
    )
    parser.add_argument(
        "--hdf5_out",
        type=str,
        required=True,
        help="Output directory for test HDF5 files",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to test convert",
    )
    args = parser.parse_args()

    task_root = Path(args.task_root)
    hdf5_out = Path(args.hdf5_out)

    ok = convert_single_task(task_root, hdf5_out, args.num_episodes)
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
