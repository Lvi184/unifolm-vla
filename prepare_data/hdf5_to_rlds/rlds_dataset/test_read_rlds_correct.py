cd /root/gpufree-data/unifolm-vla/prepare_data/hdf5_to_rlds
python -c "
import tensorflow_datasets as tfds
from rlds_dataset import rlds_dataset
ds = tfds.load('rlds_dataset', split='train', data_dir='/root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5/unifolm_rlds', shuffle_files=False)
for episode_idx, episode in enumerate(ds.take(3)):
    print(f'\n--- Episode {episode_idx} ---')
    steps = episode['steps']
    first_step = next(iter(steps.batch(1).take(1)))
    obs = first_step['observation']
    print(f'  Observation keys: {list(obs.keys())}')
    print(f'  image_primary shape: {obs[\"image_primary\"].shape}')
    if 'image_left_wrist' in obs:
        print(f'  image_left_wrist shape: {obs[\"image_left_wrist\"].shape}')
    if 'image_right_wrist' in obs:
        print(f'  image_right_wrist shape: {obs[\"image_right_wrist\"].shape}')
    print(f'  proprio shape: {obs[\"proprio\"].shape}')
    print(f'  action shape: {first_step[\"action\"].shape}')
    print(f'  language_instruction: {first_step[\"language_instruction\"].numpy().decode(\"utf-8\")}')
print('\n✅ RLDS dataset test passed!')
"