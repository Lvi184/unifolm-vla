#!/usr/bin/env python3
"""
Check that all converted HDF5 files exist and are valid.
Usage:
python check_conversion_integrity.py \
  --src_root /root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth \
  --dst_root /root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5_jpeg
"""

import argparse
import os
from pathlib import Path
import h5py
import numpy as np

from adapters.agibotworld_meta import list_all_tasks, load_task_meta

def check_task(task_root: Path, dst_root: Path):
    """Check a single task."""
    task_name = task_root.name
    out_task_dir = dst_root / task_name
    
    if not out_task_dir.exists():
        print(f"⚠️  {task_name}: output directory doesn't exist!")
        return 0, list(range(load_task_meta(task_root).get("total_episodes", 0))), 0
    
    info = load_task_meta(task_root)
    total_episodes = info.get("total_episodes", 0)
    
    missing = []
    corrupted = 0
    ok = 0
    
    for episode_id in range(total_episodes):
        out_path = out_task_dir / f"episode_{episode_id:06d}.hdf5"
        
        if not out_path.exists():
            missing.append(episode_id)
            continue
        
        # Try to open and check basic structure
        try:
            with h5py.File(out_path, "r") as f:
                # Check required groups exist
                assert "action" in f
                assert "observations" in f
                assert "images" in f["observations"]
                assert "primary" in f["observations"]["images"]
                assert "left_wrist" in f["observations"]["images"]
                assert "right_wrist" in f["observations"]["images"]
                assert "language_raw" in f
                assert "substep_reasonings" in f
                
                action = f["action"]
                assert action.ndim == 2
                episode_len = action.shape[0]
                assert episode_len > 0, f"episode has 0 steps"
                
                # Check that all images have the right length (same as action)
                qpos = f["observations"]["qpos"]
                assert qpos.shape[0] == episode_len, f"qpos length {qpos.shape[0]} != action {episode_len}"
                
                for cam in ["primary", "left_wrist", "right_wrist"]:
                    img_ds = f["observations"]["images"][cam]
                    assert img_ds.shape[0] == episode_len, f"{cam} length {img_ds.shape[0]} != action {episode_len}"
                
                substep = f["substep_reasonings"]
                assert substep.shape[0] == episode_len, f"substep_reasonings length {substep.shape[0]} != action {episode_len}"
            
            ok += 1
            
        except Exception as e:
            corrupted += 1
    
    return ok, missing, corrupted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Source root directory of agibot_data_without_depth",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        required=True,
        help="Destination root directory of converted HDF5 files",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    tasks = list_all_tasks(src_root)
    print(f"Found {len(tasks)} tasks to check\n")

    total_ok = 0
    total_total = 0
    total_missing = 0
    total_corrupted = 0
    bad_tasks = []

    for task_root in tasks:
        task_name = task_root.name
        info = load_task_meta(task_root)
        total_episodes = info.get("total_episodes", 0)
        total_total += total_episodes

        ok, missing, corrupted = check_task(task_root, dst_root)
        current_missing = len(missing)
        total_ok += ok
        total_missing += current_missing
        total_corrupted += corrupted

        if current_missing > 0 or corrupted > 0:
            bad_tasks.append((task_name, total_episodes, ok, current_missing, corrupted))
            print(f"⚠️  {task_name}: total={total_episodes}, ok={ok}, missing={current_missing}, corrupted={corrupted}")
            if missing:
                print(f"    Missing episodes: {missing[:10]}..." if len(missing) > 10 else missing)
        else:
            print(f"✅ {task_name}: all {ok} episodes OK")

    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  Total episodes: {total_total}")
    print(f"  OK:           {total_ok}")
    print(f"  Missing:      {total_missing}")
    print(f"  Corrupted:    {total_corrupted}")
    print("="*60)

    if total_missing == 0 and total_corrupted == 0:
        print("\n🎉 All episodes converted successfully!")
    else:
        print("\n⚠️  Some episodes are missing or corrupted. You need to re-run conversion to fix.")
        print("   Just re-run the same command, it will automatically convert the missing ones.")

if __name__ == "__main__":
    main()
