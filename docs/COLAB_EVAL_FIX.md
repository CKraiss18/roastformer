# Quick Fix for Evaluation Notebook Cell 10 Error

**Error**: `roastformer_data_20251118_090504` not found

**Cause**: Cell 6 has hardcoded old zip path from Nov 11, but you created a new package on Nov 18

---

## Fix: Update Cell 6

Replace the entire **Cell 6** with this:

```python
# Extract data
import zipfile
import os

print("="*80)
print("EXTRACTING DATA")
print("="*80)

# 👉 UPDATE THIS PATH to your actual zip file name
zip_path = '/content/gdrive/MyDrive/Colab Notebooks/GEN_AI/roastformer_data_20251118_090504.zip'

if os.path.exists(zip_path):
    os.chdir('/content')
    print(f"Working directory: {os.getcwd()}")

    print(f"\n📦 Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

    print("✅ Extraction complete")

    # Verify preprocessed_data exists
    if os.path.exists('preprocessed_data'):
        print("✅ Found preprocessed_data directory")
        import json
        with open('preprocessed_data/dataset_stats.json', 'r') as f:
            stats = json.load(f)
        print(f"\n📊 Dataset: {stats['total_profiles']} profiles")
    else:
        print("❌ preprocessed_data not found!")
        print("Available directories:")
        print(os.listdir('.'))
else:
    print(f"❌ Zip not found at: {zip_path}")
    print("\n📁 Available files in GEN_AI folder:")
    gen_ai_dir = '/content/gdrive/MyDrive/Colab Notebooks/GEN_AI'
    if os.path.exists(gen_ai_dir):
        files = [f for f in os.listdir(gen_ai_dir) if f.endswith('.zip')]
        for f in files:
            print(f"  - {f}")

print("="*80)
```

---

## Steps to Fix:

1. **In Colab**, replace Cell 6 with the code above
2. **Run Cell 6** - it will now extract the correct zip file
3. **If the zip name is different**, update the `zip_path` line to match your exact file name
4. **Continue to Cell 10** - should now work!

---

## Alternative: Auto-Find Latest Zip

If you're not sure of the exact name, use this version of Cell 6:

```python
# Extract data (auto-find latest zip)
import zipfile
import os
from pathlib import Path

print("="*80)
print("EXTRACTING DATA")
print("="*80)

# Auto-find latest roastformer zip file
gen_ai_dir = '/content/gdrive/MyDrive/Colab Notebooks/GEN_AI'
zip_files = sorted([f for f in os.listdir(gen_ai_dir) if f.startswith('roastformer_data') and f.endswith('.zip')])

if zip_files:
    latest_zip = zip_files[-1]  # Get most recent
    zip_path = os.path.join(gen_ai_dir, latest_zip)

    print(f"📦 Found zip file: {latest_zip}")

    os.chdir('/content')
    print(f"Working directory: {os.getcwd()}")

    print(f"\n📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

    print("✅ Extraction complete")

    # Verify preprocessed_data exists
    if os.path.exists('preprocessed_data'):
        print("✅ Found preprocessed_data directory")
        import json
        with open('preprocessed_data/dataset_stats.json', 'r') as f:
            stats = json.load(f)
        print(f"\n📊 Dataset: {stats['total_profiles']} profiles")
    else:
        print("❌ preprocessed_data not found after extraction!")
        print("Contents of /content:")
        print(os.listdir('/content'))
else:
    print(f"❌ No roastformer_data zip files found in {gen_ai_dir}")

print("="*80)
```

**Advantage**: This auto-finds the latest zip, so you don't need to update paths manually!

---

## After Fixing Cell 6:

1. Re-run cells in order: **1 → 2 → 3 → 4 → 5 → 6**
2. Verify Cell 6 outputs: `✅ Found preprocessed_data directory`
3. Continue to Cell 10 - should work now!

---

**Root Cause**: You packaged a new zip file today (Nov 18) but the notebook still references the old Nov 11 zip.

**Quick Fix**: Update zip path in Cell 6 ✅
