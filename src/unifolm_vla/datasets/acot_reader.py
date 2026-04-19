from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any

import cv2
import numpy as np
import pandas as pd


RGB_CAMERA_KEYS = [
 "observation.images.top_head",
 "observation.images.hand_left",
 "observation.images.hand_right",
]


def _read_json(path: Path) -> Dict[str, Any]:
 with open(path, "r", encoding="utf-8") as f:
 return json.load(f)


def _maybe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
 if not path.exists():
 return []
 rows = []
 with open(path, "r", encoding="utf-8") as f:
 for line in f:
 line = line.strip()
 if line:
 rows.append(json.loads(line))
 return rows


def _decode_video_all_frames(video_path: Path) -> List[np.ndarray]:
 cap = cv2.VideoCapture(str(video_path))
 frames: List[np.ndarray] = []
 try:
 while True:
 ok, frame = cap.read()
 if not ok:
 break
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 frames.append(frame)
 finally:
 cap.release()
 return frames


def _pick_instruction(
 instruction_segments: Dict[str, List[Dict[str, Any]]],
 episode_id: int,
 frame_index: int,
 fallback_text: str = "",
) -> str:
 segs = instruction_segments.get(str(episode_id), [])
 for seg in segs:
 start_idx = int(seg["start_frame_index"])
 end_idx = int(seg["end_frame_index"])
 if start_idx <= frame_index <= end_idx:
 return str(seg["instruction"])
 if segs:
 return str(segs[-1]["instruction"])
 return fallback_text


class AgiBotWorldTaskReader:
 """
 Reads one task directory like:
 open_door/
 data/
 meta/
 videos/
 and yields ACoT-style per-step samples.
 """

 def __init__(self, task_root: str | Path):
 self.task_root = Path(task_root)
 self.meta_dir = self.task_root / "meta"
 self.data_dir = self.task_root / "data"
 self.videos_dir = self.task_root / "videos"

 self.info = _read_json(self.meta_dir / "info.json")
 self.tasks = _maybe_read_jsonl(self.meta_dir / "tasks.jsonl")
 self.episodes = _maybe_read_jsonl(self.meta_dir / "episodes.jsonl")

 self.instruction_segments = self.info.get("instruction_segments", {})
 self.chunks_size = int(self.info.get("chunks_size", 1000))

 def list_episode_ids(self) -> List[int]:
 episode_ids: List[int] = []
 for ep in self.episodes:
 if "episode_index" in ep:
 episode_ids.append(int(ep["episode_index"]))
 elif "episode_id" in ep:
 episode_ids.append(int(ep["episode_id"]))
 if episode_ids:
 return sorted(set(episode_ids))

 # fallback: infer from parquet filenames
 result = []
 for p in self.data_dir.rglob("episode_*.parquet"):
 stem = p.stem
 ep_id = int(stem.split("_")[-1])
 result.append(ep_id)
 return sorted(set(result))

 def _chunk_name(self, episode_id: int) -> str:
 chunk_id = episode_id // self.chunks_size
 return f"chunk-{chunk_id:03d}"

 def _parquet_path(self, episode_id: int) -> Path:
 return self.data_dir / self._chunk_name(episode_id) / f"episode_{episode_id:06d}.parquet"

 def _video_path(self, episode_id: int, camera_key: str) -> Path:
 return (
 self.videos_dir
 / self._chunk_name(episode_id)
 / camera_key
 / f"episode_{episode_id:06d}.mp4"
 )

 def _load_episode_table(self, episode_id: int) -> pd.DataFrame:
 parquet_path = self._parquet_path(episode_id)
 if not parquet_path.exists():
 raise FileNotFoundError(f"Missing parquet: {parquet_path}")
 return pd.read_parquet(parquet_path)

 def _load_episode_videos(self, episode_id: int) -> Dict[str, List[np.ndarray]]:
 frames_by_camera: Dict[str, List[np.ndarray]] = {}
 for cam_key in RGB_CAMERA_KEYS:
 vp = self._video_path(episode_id, cam_key)
 if vp.exists():
 frames_by_camera[cam_key] = _decode_video_all_frames(vp)
 return frames_by_camera

 def iter_episode_steps(self, episode_id: int) -> Iterator[Dict[str, Any]]:
 df = self._load_episode_table(episode_id)
 frames_by_camera = self._load_episode_videos(episode_id)

 num_steps = len(df)

 for frame_index in range(num_steps):
 row = df.iloc[frame_index]

 images: Dict[str, Optional[np.ndarray]] = {}
 for cam_key in RGB_CAMERA_KEYS:
 cam_frames = frames_by_camera.get(cam_key, [])
 images[cam_key] = cam_frames[frame_index] if frame_index < len(cam_frames) else None

 # 按你的数据列名改这里
 state = np.asarray(row["observation.state"], dtype=np.float32)
 action = np.asarray(row["action"], dtype=np.float32)

 instruction = _pick_instruction(
 instruction_segments=self.instruction_segments,
 episode_id=episode_id,
 frame_index=frame_index,
 fallback_text=str(self.info.get("task_name", "")),
 )

 yield {
 "episode_id": episode_id,
 "frame_index": frame_index,
 "observation": {
 "images": {
 "top_head": images["observation.images.top_head"],
 "hand_left": images["observation.images.hand_left"],
 "hand_right": images["observation.images.hand_right"],
 },
 "state": state,
 },
 "action": action,
 "instruction": instruction,
 }

 def iter_all_steps(self) -> Iterator[Dict[str, Any]]:
 for episode_id in self.list_episode_ids():
 yield from self.iter_episode_steps(episode_id)
