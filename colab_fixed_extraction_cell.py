"""
FIXED EXTRACTION CELL - Replace your current extraction cell with this
"""

import zipfile
import os
import shutil

print("="*80)
print("GOOGLE DRIVE METHOD (FIXED)")
print("="*80)

# Paths
zip_path = '/content/gdrive/MyDrive/Colab Notebooks/GEN_AI/roastformer_data_20251111_092727.zip'
extract_to = '/content/temp_extract'  # Extract to temp location first
final_location = '/content'

# Step 1: Check zip exists
if not os.path.exists(zip_path):
    print(f"❌ Zip file not found at: {zip_path}")
    print(f"Please ensure the file is uploaded to: My Drive/Colab Notebooks/GEN_AI/")
else:
    print(f"✅ Found zip file!")

    # Step 2: Create temp extract directory
    os.makedirs(extract_to, exist_ok=True)

    # Step 3: Extract to temp location
    print(f"📦 Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Step 4: Check what was extracted
    print("\n📁 Checking extracted contents...")
    extracted_items = os.listdir(extract_to)
    print(f"Extracted {len(extracted_items)} top-level items:")
    for item in extracted_items[:10]:  # Show first 10
        print(f"  - {item}")

    # Step 5: Move files to /content/
    print("\n📦 Moving files to working directory...")

    # Check if there's a roastformer_data folder or similar
    possible_parent_dirs = [
        'roastformer_data_20251111_092727',
        'roastformer_data',
        '.'
    ]

    source_dir = None
    for parent in possible_parent_dirs:
        test_path = os.path.join(extract_to, parent, 'preprocessed_data')
        if os.path.exists(test_path):
            source_dir = os.path.join(extract_to, parent)
            print(f"✅ Found data in: {parent}/")
            break

    if source_dir is None:
        # Try direct extraction
        if os.path.exists(os.path.join(extract_to, 'preprocessed_data')):
            source_dir = extract_to
            print(f"✅ Found data at root level")

    if source_dir:
        # Copy each top-level item to /content/
        for item in os.listdir(source_dir):
            src = os.path.join(source_dir, item)
            dst = os.path.join(final_location, item)

            # Remove destination if it exists
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)

            # Copy
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            print(f"  ✅ Copied: {item}")

        # Step 6: Clean up temp directory
        shutil.rmtree(extract_to)

        # Step 7: Verify
        print("\n📊 VERIFICATION:")
        print("="*80)

        required_paths = [
            'preprocessed_data',
            'src/dataset/preprocessed_data_loader.py',
            'src/model/transformer_adapter.py',
            'train_transformer.py'
        ]

        all_good = True
        for path in required_paths:
            full_path = os.path.join(final_location, path)
            exists = os.path.exists(full_path)
            status = "✅" if exists else "❌"
            print(f"{status} {path}")
            if not exists:
                all_good = False

        if all_good:
            print("\n✅✅✅ EXTRACTION SUCCESSFUL! ✅✅✅")
            print("\npreprocessed_data contents:")
            !ls -lh /content/preprocessed_data/
        else:
            print("\n❌ Some files still missing. Showing full /content/ structure:")
            !ls -la /content/
    else:
        print("\n❌ Could not find preprocessed_data in extracted files")
        print("Showing what was extracted:")
        !ls -laR {extract_to} | head -50

print("="*80)
