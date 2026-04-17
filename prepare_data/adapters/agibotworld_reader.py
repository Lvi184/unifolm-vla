from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .agibotworld_meta import load_task_meta, get_instruction_for_frame

# RGB camera keys we know exist in dataset_without_depth
# Skip any depth cameras
KNOWN_RGB_CAMERAS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]

def read_all_video_frames(video_path: Path) -> List[np.ndarray]:
    """Read all frames from an mp4 video and convert to RGB (HWC)."""
    if not video_path.exists():
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

def load_episode(
    task_root: Path,
    info: Dict[str, Any],
    episode_id: int,
) -> List[Dict[str, Any]]:
    """
    Load a complete episode from AgiBotWorld data:
    - Reads parquet for state/action/frame_index
    - Reads all RGB videos for this episode
    - Aligns frames by index
    - Returns list of aligned steps
    """
    chunk_id = episode_id // info.get("chunks_size", 1000)
    
    # Load parquet data
    parquet_path = task_root / info["data_path"].format(
        episode_chunk=chunk_id,
        episode_index=episode_id
    )
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Load all camera videos
    camera_frames: Dict[str, List[np.ndarray]] = {}
    for cam_key in KNOWN_RGB_CAMERAS:
        video_path = task_root / info["video_path"].format(
            episode_chunk=chunk_id,
            video_key=cam_key,
            episode_index=episode_id
        )
        frames = read_all_video_frames(video_path)
        if frames:
            camera_frames[cam_key] = frames
    
    # Align frames - match parquet rows to video frames
    T = len(df)
    aligned_steps = []
    
    for i in range(T):
        row = df.iloc[i]
        frame_index = int(row["frame_index"])
        state159 = np.array(row["observation.state"], dtype=np.float32)
        action40 = np.array(row["action"], dtype=np.float32)
        
        # Get images for this frame from each camera
        images: Dict[str, np.ndarray] = {}
        for cam_key, frames in camera_frames.items():
            if frame_index < len(frames):
                # Store frame name in simpler format for later mapping
                simple_name = cam_key.replace("observation.images.", "")
                images[simple_name] = frames[frame_index]
        
        aligned_steps.append({
            "frame_index": frame_index,
            "images": images,
            "state159": state159,
            "action40": action40,
        })
    
    # Validate alignment
    for cam_key, frames in camera_frames.items():
        if len(frames) < T:
            print(f"WARNING: Episode {episode_id}: {cam_key} has {len(frames)} < {T} steps")
    
    return aligned_steps

def load_and_align_episode(
    task_root: Path,
    episode_id: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convenience: load task meta, then load and align episode."""
    info = load_task_meta(task_root)
    steps = load_episode(task_root, info, episode_id)
    return info, steps
