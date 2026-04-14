#!/usr/bin/env python
"""
Script to check how many episodes are in an HDF5 file
"""
import h5py
import sys

if len(sys.argv) < 2:
    print("Usage: python check_hdf5_episodes.py <hdf5_file>")
    sys.exit(1)

hdf5_file = sys.argv[1]

print(f"Checking HDF5 file: {hdf5_file}")
print()

try:
    with h5py.File(hdf5_file, 'r') as f:
        print("📊 HDF5 File Structure:")
        print("=" * 60)
        
        # List all top-level keys
        print("Top-level keys:")
        for key in f.keys():
            print(f"  {key}")
        
        print()
        
        # Check observations
        if 'observations' in f:
            print("Observations:")
            for obs_key in f['observations'].keys():
                print(f"  {obs_key}")
                
                # If it's a dataset, check its shape
                if isinstance(f['observations'][obs_key], h5py.Dataset):
                    print(f"    Shape: {f['observations'][obs_key].shape}")
                
                # If it's images
                if obs_key == 'images':
                    print("  Images:")
                    for img_key in f['observations']['images'].keys():
                        if isinstance(f['observations']['images'][img_key], h5py.Dataset):
                            print(f"    {img_key}: shape={f['observations']['images'][img_key].shape}")
        
        print()
        
        # Check action
        if 'action' in f:
            if isinstance(f['action'], h5py.Dataset):
                print(f"Action: shape={f['action'].shape}")
        
        print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
