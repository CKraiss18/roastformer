"""
SIMPLE FIX - Replace your extraction cell with this code
"""

import zipfile
import os

print("="*80)
print("EXTRACTING DATA FROM GOOGLE DRIVE")
print("="*80)

# Path to zip file
zip_path = '/content/gdrive/MyDrive/Colab Notebooks/GEN_AI/roastformer_data_20251111_092727.zip'

# Verify zip exists
if not os.path.exists(zip_path):
    print(f"❌ ERROR: Zip file not found!")
    print(f"Looking for: {zip_path}")
    print("\nPlease ensure roastformer_data_20251111_092727.zip is uploaded to:")
    print("My Drive/Colab Notebooks/GEN_AI/")
else:
    print(f"✅ Found zip file")

    # Change to /content directory before extracting
    os.chdir('/content')
    print(f"Working directory: {os.getcwd()}")

    # Extract
    print(f"\n📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

    print("✅ Extraction complete")

    # Verify extraction
    print("\n📁 Verifying extracted files:")
    print("="*80)

    required_paths = [
        'preprocessed_data/train_profiles.json',
        'preprocessed_data/val_profiles.json',
        'preprocessed_data/train_metadata.csv',
        'preprocessed_data/val_metadata.csv',
        'preprocessed_data/dataset_stats.json',
        'src/dataset/preprocessed_data_loader.py',
        'src/model/transformer_adapter.py',
        'train_transformer.py'
    ]

    all_good = True
    for path in required_paths:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        size = f"({os.path.getsize(path)/1024/1024:.1f} MB)" if exists else ""
        print(f"{status} {path} {size}")
        if not exists:
            all_good = False

    print("="*80)

    if all_good:
        print("\n✅✅✅ ALL FILES READY! ✅✅✅")

        # Show dataset stats
        import json
        with open('preprocessed_data/dataset_stats.json', 'r') as f:
            stats = json.load(f)

        print("\n📊 Dataset Statistics:")
        print(f"   Total profiles: {stats['total_profiles']}")
        print(f"   Training: {stats['train_size']}")
        print(f"   Validation: {stats['val_size']}")
        print(f"   Unique origins: {stats['unique_origins']}")
        print(f"   Unique processes: {stats['unique_processes']}")
        print("\n✅ Ready to train!")
    else:
        print("\n❌ SOME FILES MISSING!")
        print("\nDebugging info - contents of /content/:")
        !ls -la /content/
        print("\nIf preprocessed_data folder exists somewhere else:")
        !find /content -name "preprocessed_data" -type d 2>/dev/null

print("="*80)
