# Google Drive Upload Method for Colab

**Problem**: Zip file too large for Colab's file upload widget (gets "Maximum call stack size exceeded")

**Solution**: Use Google Drive instead (more reliable for large files)

---

## 🚀 Quick Steps

### **1. Upload to Google Drive (30 seconds)**

1. Open Google Drive: https://drive.google.com
2. Upload `roastformer_data_20251111_092727.zip` to **"My Drive"** (root folder)
3. Wait for upload to complete

---

### **2. Replace Cell 5 in Colab Notebook**

Replace the `files.upload()` cell with this:

```python
# ============================================================================
# UPLOAD DATA VIA GOOGLE DRIVE (More reliable for large files)
# ============================================================================

from google.colab import drive
import zipfile
import os

print("="*80)
print("LOADING DATA FROM GOOGLE DRIVE")
print("="*80)

# Mount Google Drive
print("\n1. Mounting Google Drive...")
drive.mount('/content/drive')
print("   ✅ Google Drive mounted!")

# Path to zip file in Google Drive (root folder)
zip_filename = 'roastformer_data_20251111_092727.zip'
zip_path = f'/content/drive/MyDrive/{zip_filename}'

print(f"\n2. Looking for: {zip_filename}")

if os.path.exists(zip_path):
    print(f"   ✅ Found zip file!")

    # Get file size
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"   📦 Size: {size_mb:.2f} MB")

    # Extract
    print(f"\n3. Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/')

    print("   ✅ Data extracted successfully!")

    # Verify extraction
    print("\n4. Verifying extracted files:")
    if os.path.exists('preprocessed_data'):
        files = os.listdir('preprocessed_data')
        for f in sorted(files):
            fpath = os.path.join('preprocessed_data', f)
            fsize = os.path.getsize(fpath) / 1024  # KB
            print(f"   ✅ {f} ({fsize:.1f} KB)")
    else:
        print("   ❌ preprocessed_data directory not found!")

    print("\n" + "="*80)
    print("✅ DATA LOADED SUCCESSFULLY!")
    print("="*80)

else:
    print(f"   ❌ Zip file not found at: {zip_path}")
    print("\n📋 Please:")
    print(f"   1. Go to https://drive.google.com")
    print(f"   2. Upload: {zip_filename}")
    print(f"   3. Make sure it's in 'My Drive' (root folder)")
    print(f"   4. Re-run this cell")
    print("="*80)
```

---

### **3. Run the Cell**

1. Click "Run cell"
2. **Authorize Google Drive** when prompted:
   - Click the authorization link
   - Choose your Google account
   - Click "Allow"
   - Paste the code back (or it happens automatically)
3. Data will extract automatically

---

## ✅ Expected Output

```
================================================================================
LOADING DATA FROM GOOGLE DRIVE
================================================================================

1. Mounting Google Drive...
   Mounted at /content/drive
   ✅ Google Drive mounted!

2. Looking for: roastformer_data_20251111_092727.zip
   ✅ Found zip file!
   📦 Size: 1.07 MB

3. Extracting data...
   ✅ Data extracted successfully!

4. Verifying extracted files:
   ✅ dataset_stats.json (0.3 KB)
   ✅ train_metadata.csv (18.4 KB)
   ✅ train_profiles.json (11160.4 KB)
   ✅ val_metadata.csv (3.2 KB)
   ✅ val_profiles.json (1835.9 KB)

================================================================================
✅ DATA LOADED SUCCESSFULLY!
================================================================================
```

---

## 🎯 Why This Works

| Method | Limit | Our File | Result |
|--------|-------|----------|--------|
| **files.upload()** | ~10 MB with complex JSON | 13 MB JSON inside | ❌ Fails |
| **Google Drive** | 15 GB | 1 MB zip | ✅ Works |

Google Drive handles large files much better than the direct upload widget.

---

## ⚠️ Troubleshooting

### "File not found"
- Make sure zip is in **root folder** of Google Drive (not in subfolders)
- Check filename exactly matches: `roastformer_data_20251111_092727.zip`

### "Permission denied"
- Re-run the cell
- Authorize Google Drive access when prompted

### "Extraction failed"
- Download the zip from Google Drive and re-upload
- Make sure upload completed (not still uploading)

---

## 🚀 After This Works

Continue with Cell 6 (verification) - should now show:
```
✅ Total profiles: 144
   Training: 123
   Validation: 21
```

Then proceed with training!

---

**This method is more reliable for large files and only requires uploading once to Google Drive.**
