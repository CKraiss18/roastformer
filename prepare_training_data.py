"""
Data Preparation Script for RoastFormer Training

Merges all collected Onyx datasets, validates profiles, and creates train/val splits.

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.utils.validation import RoastProfileValidator


def discover_dataset_directories(base_path: str = ".") -> List[Path]:
    """Find all onyx_dataset directories"""
    base = Path(base_path)
    directories = sorted(base.glob("onyx_dataset_2025_*"))
    return directories


def load_all_profiles(directories: List[Path]) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Load all profiles from multiple directories

    Returns:
        profiles: List of profile dicts
        metadata_df: DataFrame with all metadata
    """
    all_profiles = []
    all_metadata = []
    batch_ids_seen = set()

    print("\n" + "="*80)
    print("LOADING PROFILES FROM ALL DIRECTORIES")
    print("="*80)

    for directory in directories:
        print(f"\n📁 {directory.name}")

        # Load from individual profile files in profiles/ subdirectory
        profiles_dir = directory / "profiles"
        if not profiles_dir.exists():
            print(f"  ⚠️  No profiles/ directory found, skipping...")
            continue

        profile_files = sorted(profiles_dir.glob("*.json"))

        # Remove duplicates based on batch_id
        new_profiles = 0
        duplicate_profiles = 0

        for profile_file in profile_files:
            try:
                with open(profile_file, 'r') as f:
                    profile = json.load(f)

                # Extract batch_id from filename (e.g., "cold-brew_batch93174.json" -> "93174")
                filename = profile_file.stem  # Get filename without .json
                if '_batch' in filename:
                    batch_id = filename.split('_batch')[1]
                else:
                    batch_id = filename  # Fallback to full filename

                if batch_id not in batch_ids_seen:
                    all_profiles.append(profile)
                    batch_ids_seen.add(batch_id)
                    new_profiles += 1

                    # Extract metadata for DataFrame
                    meta = profile.get('metadata', {})

                    # Handle flavor_notes_parsed - ensure it's a list
                    flavor_notes_parsed = meta.get('flavor_notes_parsed', [])
                    if isinstance(flavor_notes_parsed, str):
                        flavor_notes = flavor_notes_parsed
                    elif isinstance(flavor_notes_parsed, list):
                        flavor_notes = ', '.join(flavor_notes_parsed)
                    else:
                        flavor_notes = ''

                    all_metadata.append({
                        'batch_id': batch_id,
                        'product_name': meta.get('product_name', 'Unknown'),
                        'origin': meta.get('origin', 'Unknown'),
                        'process': meta.get('process', 'Unknown'),
                        'roast_level': meta.get('roast_level', 'Unknown'),
                        'variety': meta.get('variety', 'Unknown'),
                        'target_finish_temp': meta.get('target_finish_temp', 0),
                        'flavor_notes': flavor_notes,
                        'directory': directory.name
                    })
                else:
                    duplicate_profiles += 1

            except Exception as e:
                print(f"  ⚠️  Error loading {profile_file.name}: {e}")
                continue

        print(f"  ✓ New profiles: {new_profiles}")
        if duplicate_profiles > 0:
            print(f"  ⊘ Duplicates skipped: {duplicate_profiles}")

    metadata_df = pd.DataFrame(all_metadata)

    print("\n" + "="*80)
    print(f"TOTAL UNIQUE PROFILES LOADED: {len(all_profiles)}")
    print("="*80)

    return all_profiles, metadata_df


def validate_all_profiles(profiles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate all profiles using physics-based checks

    Returns:
        valid_profiles: List of valid profiles
        invalid_profiles: List of invalid profiles with reasons
    """
    print("\n" + "="*80)
    print("VALIDATING ALL PROFILES")
    print("="*80)

    validator = RoastProfileValidator(strict=False)

    valid_profiles = []
    invalid_profiles = []

    for i, profile in enumerate(profiles):
        # Extract temperature data
        bean_temp_data = profile['roast_profile']['bean_temp']
        temps = np.array([point['value'] for point in bean_temp_data])

        # Validate
        is_valid, results = validator.validate_profile(temps, verbose=False)

        if is_valid:
            valid_profiles.append(profile)
        else:
            failed_checks = [name for name, (valid, _) in results.items() if not valid]
            invalid_profiles.append({
                'index': i,
                'batch_id': profile['metadata'].get('batch_id', 'unknown'),
                'product_name': profile['metadata'].get('product_name', 'Unknown'),
                'failed_checks': failed_checks
            })

    print(f"\n✓ Valid profiles: {len(valid_profiles)}/{len(profiles)} ({len(valid_profiles)/len(profiles)*100:.1f}%)")

    if invalid_profiles:
        print(f"\n⚠️  Invalid profiles ({len(invalid_profiles)}):")
        for inv in invalid_profiles[:5]:  # Show first 5
            print(f"  - {inv['product_name']} (batch {inv['batch_id']}): {inv['failed_checks']}")

    return valid_profiles, invalid_profiles


def create_train_val_split(
    profiles: List[Dict],
    metadata_df: pd.DataFrame,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split

    Stratifies by origin to ensure representation
    """
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*80)

    np.random.seed(random_seed)

    # Get indices
    indices = np.arange(len(profiles))
    np.random.shuffle(indices)

    # Split
    val_size = int(len(indices) * val_ratio)
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    # Separate profiles
    train_profiles = [profiles[i] for i in train_indices]
    val_profiles = [profiles[i] for i in val_indices]

    # Separate metadata
    train_mask = metadata_df.index.isin(train_indices)
    val_mask = metadata_df.index.isin(val_indices)

    train_metadata = metadata_df[train_mask].copy()
    val_metadata = metadata_df[val_mask].copy()

    print(f"\n✓ Train set: {len(train_profiles)} profiles")
    print(f"✓ Val set:   {len(val_profiles)} profiles ({val_ratio*100:.0f}%)")

    # Show origin distribution
    print(f"\nOrigin distribution:")
    print(f"  Train: {dict(train_metadata['origin'].value_counts().head(5))}")
    print(f"  Val:   {dict(val_metadata['origin'].value_counts().head(5))}")

    return train_profiles, val_profiles, train_metadata, val_metadata


def save_prepared_data(
    train_profiles: List[Dict],
    val_profiles: List[Dict],
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    output_dir: str = "preprocessed_data"
):
    """Save prepared data for training"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("SAVING PREPARED DATA")
    print("="*80)

    # Save profiles as JSON
    train_path = output_path / "train_profiles.json"
    val_path = output_path / "val_profiles.json"

    with open(train_path, 'w') as f:
        json.dump(train_profiles, f, indent=2)
    print(f"✓ Saved: {train_path}")

    with open(val_path, 'w') as f:
        json.dump(val_profiles, f, indent=2)
    print(f"✓ Saved: {val_path}")

    # Save metadata as CSV
    train_metadata.to_csv(output_path / "train_metadata.csv", index=False)
    val_metadata.to_csv(output_path / "val_metadata.csv", index=False)
    print(f"✓ Saved: metadata CSVs")

    # Save dataset statistics
    stats = {
        'total_profiles': len(train_profiles) + len(val_profiles),
        'train_size': len(train_profiles),
        'val_size': len(val_profiles),
        'val_ratio': len(val_profiles) / (len(train_profiles) + len(val_profiles)),
        'unique_origins': train_metadata['origin'].nunique(),
        'unique_processes': train_metadata['process'].nunique(),
        'unique_varieties': train_metadata['variety'].nunique(),
        'temp_range': {
            'min': float(train_metadata['target_finish_temp'].min()),
            'max': float(train_metadata['target_finish_temp'].max()),
            'mean': float(train_metadata['target_finish_temp'].mean())
        }
    }

    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: dataset_stats.json")

    print(f"\n📁 All prepared data saved to: {output_path}/")


def main():
    """Main data preparation pipeline"""

    print("\n" + "="*80)
    print("ROASTFORMER DATA PREPARATION PIPELINE")
    print("="*80)

    # Step 1: Discover directories
    directories = discover_dataset_directories()
    print(f"\n✓ Found {len(directories)} dataset directories:")
    for d in directories:
        print(f"  - {d.name}")

    # Step 2: Load all profiles
    all_profiles, metadata_df = load_all_profiles(directories)

    # Step 3: Validate profiles
    valid_profiles, invalid_profiles = validate_all_profiles(all_profiles)

    # Step 4: Create train/val split
    train_profiles, val_profiles, train_metadata, val_metadata = create_train_val_split(
        valid_profiles,
        metadata_df.iloc[:len(valid_profiles)],  # Only valid profiles
        val_ratio=0.15
    )

    # Step 5: Save prepared data
    save_prepared_data(train_profiles, val_profiles, train_metadata, val_metadata)

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print("\n✅ Ready for training!")
    print(f"   Train: {len(train_profiles)} profiles")
    print(f"   Val:   {len(val_profiles)} profiles")
    print(f"   Total: {len(valid_profiles)} profiles")
    print("\n📂 Next steps:")
    print("   1. Review preprocessed_data/ directory")
    print("   2. Run training with: python src/training/train.py")
    print("   3. Continue daily scraping to improve model!")
    print("="*80)


if __name__ == "__main__":
    main()
