
"""
Script lerobot to h5 for G2A robot (AGIBOT competition datasets).
Modified from convert_lerobot_to_hdf5_g2a.py to resize all images to 400x640.
"""
import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotDataProcessorG2A:
    def __init__(self, repo_id: str, root: str = None, image_dtype: str = "to_unit8", target_size: tuple = (400, 640)) -> None:
        self.image_dtype = image_dtype
        self.target_size = target_size  # (height, width)
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav", tolerance_s=0.5)

    def process_episode(self, episode_index: int) -> dict:
        """Process a single episode to extract camera images, state, and action for G2A robot."""
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()

        episode = defaultdict(list)
        cameras = defaultdict(list)
        # All cameras will have the same target shape
        target_height, target_width = self.target_size
        camera_shapes = {
            "top_head": (target_height, target_width),
            "hand_left": (target_height, target_width),
            "hand_right": (target_height, target_width)
        }

        for step_idx in tqdm(
            range(from_idx, to_idx), desc=f"Episode {episode_index}", position=1, leave=False, dynamic_ncols=True
        ):
            step = self.dataset[step_idx]

            # Extract camera images - G2A uses top_head, hand_left, hand_right
            image_dict = {}
            
            # Top head camera
            if "observation.images.top_head" in step:
                img = step["observation.images.top_head"]
                img_np = np.transpose((img.numpy() * 255).astype(np.uint8), (1, 2, 0))
                # Resize to target size if needed
                if img_np.shape[:2] != (target_height, target_width):
                    img_np = cv2.resize(img_np, (target_width, target_height), interpolation=cv2.INTER_AREA)
                image_dict["top_head"] = img_np
            
            # Left hand camera
            if "observation.images.hand_left" in step:
                img = step["observation.images.hand_left"]
                img_np = np.transpose((img.numpy() * 255).astype(np.uint8), (1, 2, 0))
                # Resize to target size
                img_np = cv2.resize(img_np, (target_width, target_height), interpolation=cv2.INTER_AREA)
                image_dict["hand_left"] = img_np
            
            # Right hand camera
            if "observation.images.hand_right" in step:
                img = step["observation.images.hand_right"]
                img_np = np.transpose((img.numpy() * 255).astype(np.uint8), (1, 2, 0))
                # Resize to target size
                img_np = cv2.resize(img_np, (target_width, target_height), interpolation=cv2.INTER_AREA)
                image_dict["hand_right"] = img_np

            for key, value in image_dict.items():
                if self.image_dtype == "to_unit8":
                    cameras[key].append(value)
                elif self.image_dtype == "to_bytes":
                    success, encoded_img = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    if not success:
                        raise ValueError(f"Image encoding failed for key: {key}")
                    cameras[key].append(np.void(encoded_img.tobytes()))

            # G2A state and action are already in the correct format (159D state, 40D action)
            state = step['observation.state'].numpy()
            action = step['action'].numpy()

            episode["state"].append(state)
            episode["action"].append(action)
            # For compatibility with existing code, duplicate state/action for ee_state/ee_action
            episode["ee_state"].append(state)
            episode['ee_action'].append(action)

        episode["cameras"] = cameras
        episode["task"] = step["task"]
        episode["episode_length"] = to_idx - from_idx

        # Data configuration for later use
        episode["data_cfg"] = {
            "camera_names": list(image_dict.keys()),
            "camera_shapes": camera_shapes,  # All cameras have the same target shape
            "state_dim": np.squeeze(state.shape),
            "ee_state_dim": np.squeeze(state.shape),
            "action_dim": np.squeeze(action.shape),
            "ee_action_dim": np.squeeze(action.shape),
        }
        episode["episode_index"] = episode_index

        return episode


class H5Writer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_h5(self, episode: dict) -> None:
        """Write episode data to HDF5 file."""
        episode_length = episode["episode_length"]
        episode_index = episode["episode_index"]
        state = episode["state"]
        action = episode["action"]
        ee_state = episode["ee_state"]
        ee_action = episode["ee_action"]
        qvel = np.zeros_like(episode["state"])
        cameras = episode["cameras"]
        task = episode["task"]
        data_cfg = episode["data_cfg"]

        # Prepare data dictionary
        data_dict = {
            "/observations/qpos": [state],
            "/observations/ee_qpos": [ee_state],
            "/observations/qvel": [qvel],
            "/action": [action],
            "ee_action": [ee_action],
            **{f"/observations/images/{k}": [v] for k, v in cameras.items()},
        }

        h5_path = os.path.join(self.output_dir, f"episode_{episode_index}.hdf5")

        with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as root:
            # Set attributes
            root.attrs["sim"] = False

            # Create datasets
            obs = root.create_group("observations")
            image = obs.create_group("images")

            # Write camera images - all cameras have the same target shape
            target_height, target_width = data_cfg["camera_shapes"]["top_head"]
            for cam_name, images in cameras.items():
                image.create_dataset(
                    cam_name,
                    shape=(episode_length, target_height, target_width, 3),
                    dtype="uint8",
                    chunks=(1, target_height, target_width, 3),
                    compression="gzip",
                )

            # Write state and action data
            obs.create_dataset("qpos", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            obs.create_dataset("ee_qpos", (episode_length, data_cfg["ee_state_dim"]), dtype="float32", compression="gzip")
            obs.create_dataset("qvel", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("action", (episode_length, data_cfg["action_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("ee_action", (episode_length, data_cfg["ee_action_dim"]), dtype="float32", compression="gzip")
            
            # Write metadata
            root.create_dataset("is_edited", (1,), dtype="uint8")
            substep_reasonings = root.create_dataset(
                "substep_reasonings", (episode_length,), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip"
            )
            root.create_dataset("language_raw", data=task)
            substep_reasonings[:] = [task] * episode_length

            # Write additional data
            for name, array in data_dict.items():
                root[name][...] = array


def lerobot_to_h5_g2a(repo_id: str, output_dir: Path, root: str = None, target_size: tuple = (400, 640)) -> None:
    """Main function to process and write G2A LeRobot data to HDF5 format with resized images."""
    data_processor = LeRobotDataProcessorG2A(
        repo_id, root, image_dtype="to_unit8", target_size=target_size
    )
    h5_writer = H5Writer(output_dir)

    # Process each episode
    for episode_index in tqdm(
        range(data_processor.dataset.num_episodes), desc="Episodes", position=0, dynamic_ncols=True
    ):
        if os.path.exists(os.path.join(output_dir, f"episode_{episode_index}.hdf5")):
            print(f"Episode {episode_index} already exists")
            continue
        try:
            episode = data_processor.process_episode(episode_index)
            h5_writer.write_to_h5(episode)
        except Exception as e:
            print(f"⚠️  Error converting episode {episode_index}: {e}")
            print(f"   Skipping and continuing...")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")
    parser.add_argument("--target_height", type=int, default=400, help="Target height for resized images")
    parser.add_argument("--target_width", type=int, default=640, help="Target width for resized images")
    args = parser.parse_args()
    repo_id = os.path.basename(args.data_path)
    root_path = args.data_path
    output_dir = args.target_path
    target_size = (args.target_height, args.target_width)
    lerobot_to_h5_g2a(repo_id, output_dir, root_path, target_size)
