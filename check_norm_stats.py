
#!/usr/bin/env python3
"""
Check the normalization statistics for the first 32 dimensions
and compare with our current handling.
"""

import json

# Load dataset statistics
stats_path = "/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v1/dataset_statistics.json"

with open(stats_path, 'r') as f:
    stats = json.load(f)

dataset_key = "rlds_dataset"
proprio_stats = stats[dataset_key]['proprio']
action_stats = stats[dataset_key]['action']

print("=" * 80)
print("PROPRIOCEPTION (State) Statistics - First 32 Dimensions")
print("=" * 80)
print(f"Total dimension: {len(proprio_stats['mean'])}")
print()
print("First 32 dimensions:")
print(f"{'Index':<6} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
print("-" * 80)
for i in range(32):
    mean = proprio_stats['mean'][i]
    std = proprio_stats['std'][i]
    min_val = proprio_stats['min'][i]
    max_val = proprio_stats['max'][i]
    print(f"{i:<6} {mean:<15.6f} {std:<15.6f} {min_val:<15.6f} {max_val:<15.6f}")

print()
print("=" * 80)
print("ACTION Statistics - First 32 Dimensions")
print("=" * 80)
print(f"Total dimension: {len(action_stats['mean'])}")
print()
print("First 32 dimensions:")
print(f"{'Index':<6} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
print("-" * 80)
for i in range(32):
    mean = action_stats['mean'][i]
    std = action_stats['std'][i]
    min_val = action_stats['min'][i]
    max_val = action_stats['max'][i]
    print(f"{i:<6} {mean:<15.6f} {std:<15.6f} {min_val:<15.6f} {max_val:<15.6f}")

print()
print("=" * 80)
print("KEY QUESTION: Is genie_sim's 32D the SAME as our first 32D?")
print("=" * 80)
print()
print("Let's check the 21D conversion pattern we saw before...")
print("ACoT-VLA's 21D comes from these indices in 159D:")
print("  list(range(30,44)) + [0,1] + list(range(75,80))")
print("  = [30-43] (14D) + [0-1] (2D) + [75-79] (5D)")
print()
print("But wait! genie_sim is sending 32D, NOT 21D!")
print("What is the 32D format?")
