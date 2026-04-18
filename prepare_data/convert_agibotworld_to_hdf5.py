#!/usr/bin/env python3
"""
Convert the entire AgiBotWorldChallenge-2026 Reasoning2Action-Sim dataset
to UnifoLM-VLA HDF5 format.

Follows ACoT Go2/G2A rules:
  - 159D state -> 21D control state
  - 40D action -> 21D control action
  - Keeps all RGB cameras (top_head, hand_left, hand_right) - skips depth
  - Assigns instruction per frame based on instruction_segments from info.json
  - Outputs HDF5 in the format UnifoLM hdf5_to_rlds expects

Usage:
  python convert_agibotworld_to_hdf5.py \
    --src_root /root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth \
    --dst_root /root/gpufree-data/unifolm_hdf5 \
    --workers 12

  # Auto-detect number of workers = CPU count
  python convert_agibotworld_to_hdf5.py \
    --src_root /root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth \
    --dst_root /root/gpufree-data/unifolm_hdf5

  # Don't skip existing (overwrite)
  python convert_agibotworld_to_hdf5.py \
    --src_root ... --dst_root ... --no-skip-existing
"""

import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import multiprocessing
from multiprocessing import Pool, Lock
import psutil

from adapters.g2a_space_adapter import (
    state159_to_21,
    action40_to_21,
    get_mapping_info,
)
from adapters.agibotworld_meta import (
    list_all_tasks,
    load_task_meta,
    get_instruction_for_frame,
)
from adapters.agibotworld_reader import (
    load_episode,
)

# Map AgiBot camera names to UnifoLM standard camera names
CAMERA_NAME_MAP = {
    "top_head": "primary",
    "hand_left": "left_wrist",
    "hand_right": "right_wrist",
}

def write_episode_hdf5(
    steps: list,
    info: dict,
    episode_id: int,
    out_path: Path,
) -> None:
    """Write one episode to HDF5 in UnifoLM expected format."""
    episode_len = len(steps)
    
    # Pre-allocate buffers for all expected cameras - guarantee same length as steps
    images_buffers: dict[str, list[np.ndarray]] = {
        "primary": [],
        "left_wrist": [],
        "right_wrist": [],
    }
    
    qpos = []
    qvel = []
    actions = []
    instructions = []
    
    for step in steps:
        # Project state/action to 21D
        state21 = state159_to_21(step["state159"])
        action21 = action40_to_21(step["action40"])
        instruction = get_instruction_for_frame(info, episode_id, step["frame_index"])
        
        # Map and validate images - one per step guaranteed (all three required per dataset spec)
        step_images = step["images"]
        # Primary (top_head) is required
        if "top_head" not in step_images:
            raise ValueError(f"Episode {episode_id}, frame {step['frame_index']}: missing top_head camera (required for primary)")
        images_buffers["primary"].append(step_images["top_head"])
        
        # Left wrist (hand_left) is required per dataset spec
        if "hand_left" not in step_images:
            raise ValueError(f"Episode {episode_id}, frame {step['frame_index']}: missing hand_left camera (required)")
        images_buffers["left_wrist"].append(step_images["hand_left"])
        
        # Right wrist (hand_right) is required per dataset spec
        if "hand_right" not in step_images:
            raise ValueError(f"Episode {episode_id}, frame {step['frame_index']}: missing hand_right camera (required)")
        images_buffers["right_wrist"].append(step_images["hand_right"])
        
        qpos.append(state21)
        qvel.append(np.zeros_like(state21, dtype=np.float32))
        actions.append(action21)
        instructions.append(instruction)
    
    # Stack into numpy arrays
    qpos = np.stack(qpos, axis=0).astype(np.float32)
    qvel = np.stack(qvel, axis=0).astype(np.float32)
    actions = np.stack(actions, axis=0).astype(np.float32)
    
    # Stack images - verify length matches episode_len
    images_stacked = {}
    for name, buffer in images_buffers.items():
        if not buffer:
            continue  # Skip empty buffer
        if len(buffer) != episode_len:
            raise RuntimeError(
                f"Episode {episode_id}: camera {name} has {len(buffer)} frames, expected {episode_len}"
            )
        images_stacked[name] = np.stack(buffer, axis=0).astype(np.uint8)
    
    # Get main instruction (first non-empty, or first frame instruction)
    main_instruction = next((i for i in instructions if i), instructions[0] if instructions else "")
    
    # Write HDF5
    with h5py.File(out_path, "w") as root:
        # Observations group
        obs = root.create_group("observations")
        imgs = obs.create_group("images")
        for name, data in images_stacked.items():
            imgs.create_dataset(name, data=data, compression="gzip", compression_opts=9)
        
        obs.create_dataset("qpos", data=qpos, compression="gzip")
        obs.create_dataset("qvel", data=qvel, compression="gzip")
        
        root.create_dataset("action", data=actions, compression="gzip")
        root.create_dataset("is_edited", data=np.array([0], dtype=np.uint8))
        root.create_dataset("language_raw", data=main_instruction)
        
        # Store per-frame substep reasonings = per-frame instruction
        # This matches what UnifoLM original converter does for single-task episodes
        sub = root.create_dataset(
            "substep_reasonings",
            shape=(episode_len,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
        )
        sub[:] = instructions

# Global lock for printing
print_lock = None

def init_print_lock(lock):
    global print_lock
    print_lock = lock

def convert_one_episode(args):
    """Convert a single episode - worker function for multiprocessing."""
    task_root, task_name, info, episode_id, out_task_dir, skip_existing = args
    
    out_path = out_task_dir / f"episode_{episode_id:06d}.hdf5"
    if skip_existing and out_path.exists():
        return ("skipped", episode_id, None)
    
    try:
        steps = load_episode(task_root, info, episode_id)
        if not steps:
            with print_lock:
                print(f"  WARNING: {task_name} - Episode {episode_id} is empty - skipping")
            return ("warning", episode_id, "empty")
        
        write_episode_hdf5(steps, info, episode_id, out_path)
        return ("converted", episode_id, None)
        
    except Exception as e:
        with print_lock:
            print(f"  ERROR converting {task_name} - episode {episode_id}: {str(e)}")
        return ("error", episode_id, str(e))

def convert_one_task(
    task_root: Path,
    dst_root: Path,
    skip_existing: bool = True,
    num_workers: int = 4,
) -> tuple[int, int]:
    """Convert all episodes in one task with parallel processing."""
    task_name = task_root.name
    print(f"\n=== Converting task: {task_name} (workers: {num_workers}) ===")
    out_task_dir = dst_root / task_name
    out_task_dir.mkdir(parents=True, exist_ok=True)
    
    info = load_task_meta(task_root)
    total_episodes = info.get("total_episodes", 0)
    print(f"  Total episodes: {total_episodes}")
    
    # Prepare work items
    work_items = [
        (task_root, task_name, info, episode_id, out_task_dir, skip_existing)
        for episode_id in range(total_episodes)
    ]
    
    converted = 0
    skipped = 0
    errors = 0
    
    # Process in parallel
    with Pool(processes=num_workers, initializer=init_print_lock, initargs=(Lock(),)) as pool:
        for result in pool.imap_unordered(convert_one_episode, work_items):
            status, episode_id, msg = result
            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1
            elif status == "warning":
                skipped += 1
            elif status == "error":
                errors += 1
                skipped += 1
            
            # Print progress every 20 episodes
            total_done = converted + skipped
            if total_done % 20 == 0:
                with print_lock:
                    print(f"  Progress: {total_done}/{total_episodes} (converted: {converted}, skipped: {skipped}, errors: {errors})")
    
    print(f"  Done: converted {converted}, skipped {skipped}, errors {errors}")
    return converted, skipped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Root directory of agibot_data_without_depth (contains multiple task subdirs)",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        required=True,
        help="Output root directory for converted HDF5 files",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip existing HDF5 files (overwrite)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect CPU count, use 0 for single-process)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Only convert a specific task name (for distributed processing across multiple servers)",
    )
    args = parser.parse_args()
    
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    skip_existing = not args.no_skip_existing
    
    # Auto-detect number of workers
    if args.workers is None:
        # Use 75% of available CPUs to leave some headroom
        total_cpus = psutil.cpu_count(logical=True) or multiprocessing.cpu_count()
        num_workers = max(1, int(total_cpus * 0.70))
    elif args.workers <= 0:
        num_workers = 1
    else:
        num_workers = args.workers
    
    # Print mapping info for verification
    mapping_info = get_mapping_info()
    print("=== G2A Space Mapping ===")
    print(f"Output dimensions: state={mapping_info['state_dim']}, action={mapping_info['action_dim']}")
    print(f"Parallel workers: {num_workers}")
    print(f"Skip existing: {skip_existing}")
    print()
    
    # Find all tasks
    tasks = list_all_tasks(src_root)
    print(f"Found {len(tasks)} valid tasks:")
    for task in tasks:
        print(f"  - {task.name}")
    print()
    
    # Filter tasks if --task specified (for distributed processing)
    if args.task is not None:
        tasks = [t for t in tasks if t.name == args.task]
        if not tasks:
            print(f"ERROR: Task '{args.task}' not found!")
            return
        print(f"⚠️  Only converting single task: {args.task}")
        print()
    
    # Convert each task
    total_converted = 0
    total_skipped = 0
    
    if num_workers > 1:
        # Parallel episode conversion within each task
        for task_root in tasks:
            c, s = convert_one_task(task_root, dst_root, skip_existing, num_workers)
            total_converted += c
            total_skipped += s
    else:
        # Original single-threaded mode
        for task_root in tasks:
            # Legacy single-process conversion
            converted = 0
            skipped = 0
            task_name = task_root.name
            print(f"\n=== Converting task: {task_name} (single process) ===")
            out_task_dir = dst_root / task_name
            out_task_dir.mkdir(parents=True, exist_ok=True)
            
            info = load_task_meta(task_root)
            total_episodes = info.get("total_episodes", 0)
            print(f"  Total episodes: {total_episodes}")
            
            for episode_id in range(total_episodes):
                out_path = out_task_dir / f"episode_{episode_id:06d}.hdf5"
                if skip_existing and out_path.exists():
                    skipped += 1
                    continue
                
                try:
                    steps = load_episode(task_root, info, episode_id)
                    if not steps:
                        print(f"  WARNING: Episode {episode_id} is empty - skipping")
                        continue
                    
                    write_episode_hdf5(steps, info, episode_id, out_path)
                    converted += 1
                    
                    if (converted + skipped) % 50 == 0:
                        print(f"  Progress: {converted + skipped}/{total_episodes}")
                        
                except Exception as e:
                    print(f"  ERROR converting episode {episode_id}: {str(e)}")
                    continue
            
            print(f"  Done: converted {converted}, skipped {skipped}")
            total_converted += converted
            total_skipped += skipped
    
    print("\n=== All tasks converted ===")
    print(f"Total: converted {total_converted}, skipped {total_skipped}")
    print(f"Output root: {dst_root}")
    print("\n📋 Distributed processing tips for multiple servers:")
    print("  - Use --workers N to set number of parallel processes")
    print("  - Use --task TASK_NAME to convert only one specific task per server")
    print("  - Existing converted episodes are automatically skipped")
    print("  - Next step: run hdf5_to_rlds build as instructed in UnifoLM-VLA README")

if __name__ == "__main__":
    main()
