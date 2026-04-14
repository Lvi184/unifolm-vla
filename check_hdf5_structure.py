#!/usr/bin/env python
"""Check HDF5 file structure"""
import h5py
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python check_hdf5_structure.py <hdf5_file>")
    sys.exit(1)

hdf5_file = Path(sys.argv[1])

if not hdf5_file.exists():
    print(f"File not found: {hdf5_file}")
    sys.exit(1)

print(f"Checking: {hdf5_file}")
print(f"Size: {hdf5_file.stat().st_size / (1024*1024):.2f} MB")
print()

with h5py.File(hdf5_file, 'r') as f:
    print("=== File structure ===")
    def print_attrs(name, obj):
        print(name)
        if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
            for key, val in obj.attrs.items():
                print(f"  {key}: {val}")
    
    f.visititems(print_attrs)
    
    print()
    print("=== Data shapes ===")
    if 'action' in f:
        print(f"action: {f['action'].shape}")
    if 'observations' in f:
        if 'qpos' in f['observations']:
            print(f"observations/qpos: {f['observations/qpos'].shape}")
        if 'images' in f['observations']:
            for cam in f['observations/images']:
                print(f"observations/images/{cam}: {f['observations/images'][cam].shape}")
