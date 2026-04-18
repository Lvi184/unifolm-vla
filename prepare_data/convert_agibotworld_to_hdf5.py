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
import cv2
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

def encode_jpeg_bytes(img: np.ndarray, jpeg_quality: int = 90) -> np.ndarray:
    """
    Encode RGB uint8 image [H, W, 3] to JPEG bytes stored as 1D uint8 array.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {img.shape}")

    # OpenCV encodes BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return enc

def write_episode_hdf5(
    steps: list,
    info: dict,
    episode_id: int,
    out_path: Path,
    jpeg_quality: int = 90,
) -> None:
    """
    Write one episode to HDF5 in UnifoLM expected format,
    but store images as JPEG bytes instead of raw uint8 tensors.
    This drastically reduces HDF5 file size.
    """
    episode_len = len(steps)
    if episode_len == 0:
        raise ValueError(f"Episode {episode_id} is empty")

    qpos = np.empty((episode_len, 21), dtype=np.float32)
    qvel = np.zeros((episode_len, 21), dtype=np.float32)
    actions = np.empty((episode_len, 21), dtype=np.float32)
    instructions = [""] * episode_len

    # Each frame is stored as 1D uint8 JPEG bytes
    images_buffers: dict[str, list[np.ndarray]] = {
        "primary": [],
        "left_wrist": [],
        "right_wrist": [],
    }

    # Optional: keep original HWC shapes as attrs
    first_step = steps[0]
    first_images = first_step["images"]
    if "top_head" not in first_images:
        raise ValueError(f"Episode {episode_id}: missing top_head camera (required)")
    if "hand_left" not in first_images:
        raise ValueError(f"Episode {episode_id}: missing hand_left camera (required)")
    if "hand_right" not in first_images:
        raise ValueError(f"Episode {episode_id}: missing hand_right camera (required)")

    image_shapes = {
        "primary": first_images["top_head"].shape,
        "left_wrist": first_images["hand_left"].shape,
        "right_wrist": first_images["hand_right"].shape,
    }

    for i, step in enumerate(steps):
        state21 = state159_to_21(step["state159"])
        action21 = action40_to_21(step["action40"])
        instruction = get_instruction_for_frame(info, episode_id, step["frame_index"])

        step_images = step["images"]

        if "top_head" not in step_images:
            raise ValueError(f"Episode {episode_id}, frame {step['frame_index']}: missing top_head")
        if "hand_left" not in step_images:
            raise ValueError(f"Episode {episode_id}, frame {step['frame_index']}: missing hand_left")
        if "hand_right" not in step_images:
            raise ValueError(f"Episode {episode_id}, frame {step['frame_index']}: missing hand_right")

        qpos[i] = state21
        actions[i] = action21
        instructions[i] = instruction

        images_buffers["primary"].append(
            encode_jpeg_bytes(step_images["top_head"], jpeg_quality=jpeg_quality)
        )
        images_buffers["left_wrist"].append(
            encode_jpeg_bytes(step_images["hand_left"], jpeg_quality=jpeg_quality)
        )
        images_buffers["right_wrist"].append(
            encode_jpeg_bytes(step_images["hand_right"], jpeg_quality=jpeg_quality)
        )

    main_instruction = next((x for x in instructions if x), instructions[0] if instructions else "")

    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(out_path, "w") as root:
        obs = root.create_group("observations")
        imgs = obs.create_group("images")

        for name, buffer in images_buffers.items():
            ds = imgs.create_dataset(
                name,
                shape=(episode_len,),
                dtype=vlen_uint8,
            )
            for i, enc in enumerate(buffer):
                ds[i] = enc

            ds.attrs["encoding"] = "jpeg"
            ds.attrs["jpeg_quality"] = int(jpeg_quality)
            ds.attrs["original_shape"] = image_shapes[name]

        obs.create_dataset("qpos", data=qpos, compression="gzip", compression_opts=1)
        obs.create_dataset("qvel", data=qvel, compression="gzip", compression_opts=1)
        root.create_dataset("action", data=actions, compression="gzip", compression_opts=1)
        root.create_dataset("is_edited", data=np.array([0], dtype=np.uint8))
        root.create_dataset("language_raw", data=main_instruction)

        sub = root.create_dataset(
            "substep_reasonings",
            shape=(episode_len,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        sub[:] = instructions

# Global lock for printing
print_lock = None

def init_print_lock(lock):
    global print_lock
    print_lock = lock

def convert_one_episode(args):
    """Convert a single episode - worker function for multiprocessing."""
    task_root, task_name, info, episode_id, out_task_dir, skip_existing, jpeg_quality = args
    
    out_path = out_task_dir / f"episode_{episode_id:06d}.hdf5"
    if skip_existing and out_path.exists():
        return ("skipped", episode_id, None)
    
    try:
        steps = load_episode(task_root, info, episode_id)
        if not steps:
            print_lock.acquire()
            print(f"  WARNING: {task_name} - Episode {episode_id} is empty - skipping")
            print_lock.release()
            return ("warning", episode_id, "empty")
        
        write_episode_hdf5(steps, info, episode_id, out_path, jpeg_quality=jpeg_quality)
        return ("converted", episode_id, None)
        
    except Exception as e:
        print_lock.acquire()
        print(f"  ERROR converting {task_name} - episode {episode_id}: {str(e)}")
        print_lock.release()
        return ("error", episode_id, str(e))

def convert_one_task(
    task_root: Path,
    dst_root: Path,
    skip_existing: bool = True,
    num_workers: int = 4,
    jpeg_quality: int = 90,
) -> tuple[int, int]:
    """Convert all episodes in one task with parallel processing."""
    task_name = task_root.name
    print(f"\n=== Converting task: {task_name} (workers: {num_workers}, JPEG quality: {jpeg_quality}) ===")
    out_task_dir = dst_root / task_name
    out_task_dir.mkdir(parents=True, exist_ok=True)
    
    info = load_task_meta(task_root)
    total_episodes = info.get("total_episodes", 0)
    print(f"  Total episodes: {total_episodes}")
    
    # Prepare work items
    work_items = [
        (task_root, task_name, info, episode_id, out_task_dir, skip_existing, jpeg_quality)
        for episode_id in range(total_episodes)
    ]
    
    converted = 0
    skipped = 0
    errors = 0
    
    # Process in parallel
    lock = Lock()
    with Pool(processes=num_workers, initializer=init_print_lock, initargs=(lock,)) as pool:
        # Use chunksize=4 to reduce IPC overhead
        for result in pool.imap_unordered(convert_one_episode, work_items, chunksize=4):
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
                lock.acquire()
                print(f"  Progress: {total_done}/{total_episodes} (converted: {converted}, skipped: {skipped}, errors: {errors})")
                lock.release()
    
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
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for storing images inside HDF5 (1-100). Higher = better quality but larger files.",
    )
    args = parser.parse_args()
    
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    skip_existing = not args.no_skip_existing
    jpeg_quality = args.jpeg_quality
    
    # Auto-detect number of workers
    if args.workers is None:
        # For I/O bound image conversion, use fewer workers to avoid disk contention
        # Disk I/O is usually the bottleneck, not CPU - even with many CPU cores
        total_cpus = psutil.cpu_count(logical=True) or multiprocessing.cpu_count()
        # Cap at 16 workers for large CPU count machines to avoid disk congestion
        # If you have very fast NVMe SSD, you can increase this to 32
        num_workers = min(total_cpus, 16)
        num_workers = max(1, int(num_workers))
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
    print(f"JPEG quality for images: {jpeg_quality} (images stored as JPEG bytes)")
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
            c, s = convert_one_task(task_root, dst_root, skip_existing, num_workers, jpeg_quality)
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
                    
                    write_episode_hdf5(steps, info, episode_id, out_path, jpeg_quality=jpeg_quality)
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
