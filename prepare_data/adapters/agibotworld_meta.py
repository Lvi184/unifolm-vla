import json
from pathlib import Path
from typing import Any, Dict, Optional

def load_task_meta(task_root: Path) -> Dict[str, Any]:
    """Load meta information from AgiBotWorld task directory.
    
    Reads:
    - meta/info.json: task-level metadata (total_episodes, data/video paths, etc.)
    - meta/tasks.jsonl: optional task descriptions
    - meta/episodes.jsonl: optional episode-level metadata
    """
    meta_dir = task_root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"meta directory not found at {meta_dir}")
    
    info_json = meta_dir / "info.json"
    if not info_json.exists():
        raise FileNotFoundError(f"info.json not found at {info_json}")
    
    with open(info_json, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    # Load optional metadata
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        tasks = []
        with open(tasks_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        info["tasks"] = tasks
    
    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        episodes = []
        with open(episodes_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        info["episodes"] = episodes
    
    return info

def get_instruction_for_frame(
    info: Dict[str, Any],
    episode_id: int,
    frame_index: int
) -> str:
    """Get the correct instruction for a given frame based on instruction segments.
    
    If no instruction_segments found or frame doesn't match any segment,
    falls back to the last segment in the episode.
    """
    instruction_segments = info.get("instruction_segments", {})
    segs = instruction_segments.get(str(episode_id), [])
    
    if not segs:
        # No segments available - try to get from tasks list if exists
        if "tasks" in info and len(info["tasks"]) > 0:
            return info["tasks"][0].get("name", "")
        # Fallback to task_name or empty string
        return info.get("task_name", "")
    
    for seg in segs:
        start = seg.get("start_frame_index", 0)
        end = seg.get("end_frame_index", float("inf"))
        if start <= frame_index <= end:
            return seg.get("instruction", "")
    
    # If we get here, frame is after all segments - return the last one
    return segs[-1].get("instruction", "")

def list_all_tasks(src_root: Path) -> list[Path]:
    """List all valid AgiBotWorld tasks in the source root directory.
    
    A valid task must have: meta/info.json, data/, videos/ directories.
    """
    tasks = []
    for task_dir in sorted(src_root.iterdir()):
        if not task_dir.is_dir():
            continue
        # Check required directories
        if (
            (task_dir / "meta").exists() and
            (task_dir / "data").exists() and
            (task_dir / "videos").exists() and
            (task_dir / "meta" / "info.json").exists()
        ):
            tasks.append(task_dir)
    return tasks

def verify_task_structure(task_root: Path) -> bool:
    """Verify that a task directory has the expected structure."""
    required = [
        task_root / "meta" / "info.json",
        task_root / "data",
        task_root / "videos",
    ]
    for path in required:
        if not path.exists():
            return False
    return True
