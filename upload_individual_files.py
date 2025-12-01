"""
Create individual file uploads for Colab (avoids large zip issue)

Instead of one big zip, creates smaller uploads.
"""

import zipfile
from pathlib import Path
import shutil

def create_split_packages():
    """Create smaller packages to avoid Colab upload limits"""

    print("="*80)
    print("CREATING SPLIT PACKAGES FOR COLAB")
    print("="*80)

    # Package 1: Code only (small)
    print("\n📦 Package 1: Code (roastformer_code.zip)")
    with zipfile.ZipFile('roastformer_code.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        files = [
            'src/dataset/__init__.py',
            'src/dataset/preprocessed_data_loader.py',
            'src/model/__init__.py',
            'src/model/transformer_adapter.py',
            'train_transformer.py',
        ]

        for f in files:
            if Path(f).exists():
                zipf.write(f, f)
                print(f"  ✅ {f}")

        # Add __init__ files
        zipf.writestr('src/__init__.py', '# RoastFormer\n')
        zipf.writestr('src/training/__init__.py', '# Training\n')
        zipf.writestr('src/utils/__init__.py', '# Utils\n')

    code_size = Path('roastformer_code.zip').stat().st_size / 1024
    print(f"  Size: {code_size:.1f} KB")

    # Package 2: Training data (medium)
    print("\n📦 Package 2: Training Data (roastformer_train.zip)")
    with zipfile.ZipFile('roastformer_train.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('preprocessed_data/train_profiles.json', 'preprocessed_data/train_profiles.json')
        zipf.write('preprocessed_data/train_metadata.csv', 'preprocessed_data/train_metadata.csv')
        print(f"  ✅ train_profiles.json")
        print(f"  ✅ train_metadata.csv")

    train_size = Path('roastformer_train.zip').stat().st_size / 1024 / 1024
    print(f"  Size: {train_size:.2f} MB")

    # Package 3: Validation data (small)
    print("\n📦 Package 3: Validation Data (roastformer_val.zip)")
    with zipfile.ZipFile('roastformer_val.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('preprocessed_data/val_profiles.json', 'preprocessed_data/val_profiles.json')
        zipf.write('preprocessed_data/val_metadata.csv', 'preprocessed_data/val_metadata.csv')
        zipf.write('preprocessed_data/dataset_stats.json', 'preprocessed_data/dataset_stats.json')
        print(f"  ✅ val_profiles.json")
        print(f"  ✅ val_metadata.csv")
        print(f"  ✅ dataset_stats.json")

    val_size = Path('roastformer_val.zip').stat().st_size / 1024 / 1024
    print(f"  Size: {val_size:.2f} MB")

    print("\n" + "="*80)
    print("SPLIT PACKAGES CREATED")
    print("="*80)
    print(f"1. roastformer_code.zip  ({code_size:.1f} KB)")
    print(f"2. roastformer_train.zip ({train_size:.2f} MB)")
    print(f"3. roastformer_val.zip   ({val_size:.2f} MB)")
    print("\nTotal: {:.2f} MB".format(code_size/1024 + train_size + val_size))

    print("\n📋 In Colab, upload these 3 files separately")
    print("   (Smaller files = less likely to hit upload limits)")

if __name__ == "__main__":
    create_split_packages()
