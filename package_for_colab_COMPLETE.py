"""
Package Complete RoastFormer for Google Colab
Includes: data, code, normalized loader, all scripts
"""
import zipfile
import os
from datetime import datetime
from pathlib import Path

print("="*80)
print("PACKAGING COMPLETE ROASTFORMER FOR COLAB")
print("="*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
package_name = f'roastformer_COMPLETE_{timestamp}.zip'

# Files to include
files_to_package = [
    # Data
    ('preprocessed_data/train_profiles.json', 'preprocessed_data/train_profiles.json'),
    ('preprocessed_data/val_profiles.json', 'preprocessed_data/val_profiles.json'),
    ('preprocessed_data/train_metadata.csv', 'preprocessed_data/train_metadata.csv'),
    ('preprocessed_data/val_metadata.csv', 'preprocessed_data/val_metadata.csv'),
    ('preprocessed_data/dataset_stats.json', 'preprocessed_data/dataset_stats.json'),

    # Source code - Dataset loaders (BOTH versions)
    ('src/dataset/__init__.py', 'src/dataset/__init__.py'),
    ('src/dataset/preprocessed_data_loader.py', 'src/dataset/preprocessed_data_loader.py'),
    ('src/dataset/preprocessed_data_loader_NORMALIZED.py', 'src/dataset/preprocessed_data_loader_NORMALIZED.py'),  # ✅ KEY!

    # Source code - Model
    ('src/model/__init__.py', 'src/model/__init__.py'),
    ('src/model/transformer_adapter.py', 'src/model/transformer_adapter.py'),

    # Training script
    ('train_transformer.py', 'train_transformer.py'),
]

print(f"\nCreating package: {package_name}\n")

with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for local_path, archive_path in files_to_package:
        if os.path.exists(local_path):
            zipf.write(local_path, archive_path)
            size_kb = os.path.getsize(local_path) / 1024
            print(f"✅ {archive_path:60s} ({size_kb:>8.1f} KB)")
        else:
            print(f"⚠️  MISSING: {local_path}")

# Get package size
package_size_mb = os.path.getsize(package_name) / 1024 / 1024

print(f"\n{'='*80}")
print(f"✅ Package created: {package_name}")
print(f"   Size: {package_size_mb:.2f} MB")
print(f"{'='*80}")

print("\n📋 UPLOAD INSTRUCTIONS:")
print("1. Upload this zip to Google Drive:")
print(f"   /MyDrive/Colab Notebooks/GEN_AI/{package_name}")
print("\n2. Update notebook to use this zip:")
print(f"   zip_path = '.../GEN_AI/{package_name}'")
print("\n3. Run comprehensive experiments!")
print("="*80)
