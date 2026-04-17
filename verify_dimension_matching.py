
#!/usr/bin/env python3
"""
Verify if genie_sim's 32D matches the ACoT-VLA pattern.
Compare:
- ACoT-VLA's norm_stats.json (32D, but first 21D is what matters)
- UnifoLM-VLA's dataset_statistics.json (159D) with indices:
  state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
  action_indices for 40D → 21D: [16-30] + [0-2] + [33-38]
"""

import json

# Load both stats
unifolm_stats_path = "/root/gpufree-data/unifolm-vla/results/unifolm_vla_agibot_v1/dataset_statistics.json"
acot_stats_path = "/root/gpufree-data/ACoT-VLA/norm_stats.json"

with open(unifolm_stats_path, 'r') as f:
    unifolm_stats = json.load(f)

with open(acot_stats_path, 'r') as f:
    acot_stats = json.load(f)

unifolm_key = "rlds_dataset"
u_state = unifolm_stats[unifolm_key]['proprio']
u_action = unifolm_stats[unifolm_key]['action']

a_state = acot_stats['norm_stats']['state']
a_action = acot_stats['norm_stats']['actions']

print("=" * 80)
print("COMPARING STATE DIMENSIONS")
print("=" * 80)
print()

# ACoT-VLA's 21D comes from these indices in 159D:
state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
print(f"ACoT-VLA state indices from 159D: {state_indices}")
print(f"Length: {len(state_indices)}")
print()

print(f"{'Index':<6} {'ACoT 32D':<15} {'UnifoLM 159D→21D':<15} {'Match?'}")
print("-" * 80)

for i in range(21):
    acot_mean = a_state['mean'][i]
    acot_std = a_state['std'][i]
    
    u_idx = state_indices[i]
    u_mean = u_state['mean'][u_idx]
    u_std = u_state['std'][u_idx]
    
    # Check if they roughly match
    mean_match = abs(acot_mean - u_mean) < 0.1
    std_match = abs(acot_std - u_std) < 0.1
    match = "✓" if mean_match and std_match else "✗"
    
    print(f"{i:<6} {acot_mean:<15.6f} {u_mean:<15.6f} {match}")

print()
print("=" * 80)
print("COMPARING ACTION DIMENSIONS")
print("=" * 80)
print()

# ACoT-VLA's 40D → 21D conversion:
# data["actions"][:, 16:30] + data["actions"][:, 0:2] + data["actions"][:, 33:38]
action_indices = list(range(16, 30)) + list(range(0, 2)) + list(range(33, 38))
print(f"ACoT-VLA action indices from 40D: {action_indices}")
print(f"Length: {len(action_indices)}")
print()

print(f"{'Index':<6} {'ACoT 32D':<15} {'UnifoLM 40D→21D':<15} {'Match?'}")
print("-" * 80)

for i in range(21):
    acot_mean = a_action['mean'][i]
    acot_std = a_action['std'][i]
    
    u_idx = action_indices[i]
    u_mean = u_action['mean'][u_idx]
    u_std = u_action['std'][u_idx]
    
    mean_match = abs(acot_mean - u_mean) < 0.1
    std_match = abs(acot_std - u_std) < 0.1
    match = "✓" if mean_match and std_match else "✗"
    
    print(f"{i:<6} {acot_mean:<15.6f} {u_mean:<15.6f} {match}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("If we see a lot of matches (✓), that means:")
print("1. ACoT-VLA's 21D format comes from UnifoLM's 159D/40D via those indices")
print("2. But genie_sim sends 32D, which is the 21D + 11 padding zeros!")
print()
print("The problem:")
print("- We trained UnifoLM-VLA on FULL 159D/40D")
print("- But we should have trained it on the CONVERTED 21D format like ACoT-VLA!")
print("- OR we need to do the reverse conversion at inference time!")
