"""
Package RoastFormer Data for Google Colab

Creates a zip file containing:
- Preprocessed data (train/val profiles and metadata)
- Source code (data loaders and model)
- Training scripts

This package can be uploaded to Colab for GPU training.

Author: Charlee Kraiss
Date: November 2024
"""

import zipfile
import os
from pathlib import Path
from datetime import datetime
import json

def package_for_colab():
    """Create a Colab-ready data package"""

    print("="*80)
    print("PACKAGING ROASTFORMER FOR GOOGLE COLAB")
    print("="*80)

    # Create package name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    package_name = f'roastformer_data_{timestamp}.zip'

    # Files to include
    files_to_package = [
        # Preprocessed data
        'preprocessed_data/train_profiles.json',
        'preprocessed_data/val_profiles.json',
        'preprocessed_data/train_metadata.csv',
        'preprocessed_data/val_metadata.csv',
        'preprocessed_data/dataset_stats.json',

        # Source code - data loading
        'src/dataset/__init__.py',
        'src/dataset/preprocessed_data_loader.py',

        # Source code - model
        'src/model/__init__.py',
        'src/model/transformer_adapter.py',

        # Training scripts
        'train_transformer.py',

        # Create placeholder __init__ files
    ]

    print("\n📦 Creating package...")
    print(f"   Package name: {package_name}")

    # Create the zip file
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:

        # Add all files
        for file_path in files_to_package:
            if os.path.exists(file_path):
                zipf.write(file_path, file_path)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"   ✅ {file_path} ({size_kb:.1f} KB)")
            else:
                print(f"   ⚠️  Missing: {file_path}")

        # Create __init__.py files if they don't exist
        init_files = [
            'src/__init__.py',
            'src/dataset/__init__.py',
            'src/model/__init__.py',
            'src/training/__init__.py',
            'src/utils/__init__.py'
        ]

        for init_file in init_files:
            zipf.writestr(init_file, '# RoastFormer module\n')

        # Add a README for Colab
        readme = """# RoastFormer Data Package

This package contains everything needed to train RoastFormer on Google Colab.

## Contents

### Preprocessed Data
- train_profiles.json - Training roast profiles
- val_profiles.json - Validation roast profiles
- train_metadata.csv - Training metadata
- val_metadata.csv - Validation metadata
- dataset_stats.json - Dataset statistics

### Source Code
- src/dataset/preprocessed_data_loader.py - Data loading
- src/model/transformer_adapter.py - Transformer model
- train_transformer.py - Training script

## How to Use

1. Upload this zip file to Google Colab
2. Extract: `!unzip roastformer_data_*.zip`
3. Open RoastFormer_Colab_Training.ipynb
4. Run all cells

## Dataset Info

See dataset_stats.json for:
- Number of profiles
- Train/val split
- Feature dimensions
- Unique values per feature

Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """

"""
        zipf.writestr('README.txt', readme)
        print(f"   ✅ README.txt")

    # Get package size
    package_size_mb = os.path.getsize(package_name) / 1024 / 1024

    print("\n" + "="*80)
    print("PACKAGE CREATED SUCCESSFULLY")
    print("="*80)
    print(f"📦 Package: {package_name}")
    print(f"📊 Size: {package_size_mb:.2f} MB")

    # Load and display dataset stats
    if os.path.exists('preprocessed_data/dataset_stats.json'):
        with open('preprocessed_data/dataset_stats.json', 'r') as f:
            stats = json.load(f)

        print(f"\n📈 Dataset Summary:")
        print(f"   Total profiles: {stats['total_profiles']}")
        print(f"   Training: {stats['train_size']}")
        print(f"   Validation: {stats['val_size']}")
        print(f"   Unique origins: {stats.get('unique_origins', 'N/A')}")
        print(f"   Unique processes: {stats.get('unique_processes', 'N/A')}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Go to Google Colab: https://colab.research.google.com")
    print("2. Upload RoastFormer_Colab_Training.ipynb")
    print(f"3. When prompted, upload: {package_name}")
    print("4. Enable GPU: Runtime → Change runtime type → GPU")
    print("5. Run all cells")
    print("\nEstimated training time: 30-60 minutes (with GPU)")
    print("="*80)

    return package_name


if __name__ == "__main__":
    package_name = package_for_colab()
    print(f"\n✅ Ready for Colab! Package: {package_name}")
