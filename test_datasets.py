"""
Quick test to show what the data prep will discover
Run this first to see what datasets you have
"""

import glob
from pathlib import Path

print("="*60)
print("CHECKING FOR ONYX DATASETS")
print("="*60)

# Find all onyx_dataset_* directories
pattern = 'onyx_dataset_*'
found_dirs = glob.glob(pattern)

# Filter to only directories
found_dirs = [d for d in found_dirs if Path(d).is_dir()]

if not found_dirs:
    print("\n❌ NO DATASETS FOUND!")
    print("\nPlease run:")
    print("  python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py")
    print("\nOr check your current directory:")
    import os
    print(f"  Current dir: {os.getcwd()}")
else:
    # Sort by date
    found_dirs.sort(reverse=True)
    
    print(f"\n✅ Found {len(found_dirs)} dataset(s):\n")
    
    for i, d in enumerate(found_dirs, 1):
        dir_path = Path(d)
        print(f"{i}. {dir_path.name}")
        
        # Check for CSV
        csv_path = dir_path / 'dataset_summary.csv'
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"   ✓ CSV found: {len(df)} products")
        else:
            print(f"   ✗ No CSV found")
        
        # Check for profiles
        profiles_dir = dir_path / 'profiles'
        if profiles_dir.exists():
            json_files = list(profiles_dir.glob('*.json'))
            print(f"   ✓ Profiles: {len(json_files)} JSON files")
        else:
            print(f"   ✗ No profiles/ directory")
        
        print()
    
    print("="*60)
    print("✅ READY TO RUN DATA PREPARATION")
    print("="*60)
    print("\nRun: python 01_data_preparation.py")
    print("\nThis will:")
    print("  • Load ALL datasets above")
    print("  • Remove duplicate batches")
    print("  • Combine into single training set")
    print("  • Create train/val split")
    print("  • Save to preprocessed_data/")
