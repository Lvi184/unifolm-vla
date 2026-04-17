import tensorflow_datasets as tfds

ds = tfds.load(
    "rlds_dataset",
    data_dir="/root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5/unifolm_rlds",
    split="train",
)

episode = next(iter(ds))

print("Top-level keys:", episode.keys())

steps_ds = episode["steps"]
episode_length = sum(1 for _ in steps_ds)
print("Episode length:", episode_length)

steps_ds = episode["steps"]
first_step = tfds.as_numpy(next(iter(steps_ds)))

print("Step keys:", first_step.keys())
print("Observation keys:", first_step["observation"].keys())
print("Action shape:", first_step["action"].shape)
print("Proprio shape:", first_step["observation"]["proprio"].shape)

if "image_primary" in first_step["observation"]:
    print("Primary image shape:", first_step["observation"]["image_primary"].shape)
if "image_left_wrist" in first_step["observation"]:
    print("Left wrist shape:", first_step["observation"]["image_left_wrist"].shape)
if "image_right_wrist" in first_step["observation"]:
    print("Right wrist shape:", first_step["observation"]["image_right_wrist"].shape)

print("Instruction:", first_step["language_instruction"])

episode_metadata = tfds.as_numpy(episode["episode_metadata"])
print("Metadata:", episode_metadata)