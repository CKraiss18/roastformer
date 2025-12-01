# 🎉 FULLY AUTOMATIC ADDITIVE BUILDER - v3.1

## ✅ **COMPLETE - Ready to Use!**

**File:** `onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py`

---

## 🚀 **How It Works**

### **Automatic Date-Stamped Directories**

```bash
# Oct 28 scrape
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
# Creates: onyx_dataset_2025_10_28/

# Nov 3 scrape (run same command!)
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
# Creates: onyx_dataset_2025_11_03/

# Nov 10 scrape (run same command!)
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
# Creates: onyx_dataset_2025_11_10/
```

**NO editing needed!** Directory name automatically includes today's date.

---

## 🎯 **Key Features**

### **1. Automatic Date Stamping**
- Each run creates: `onyx_dataset_YYYY_MM_DD/`
- No manual editing required
- Never overwrites previous scrapes

### **2. Global Batch History Tracking**
- Loads batch info from ALL previous scrapes
- Checks if you already have a specific batch
- Skips duplicate batches automatically

### **3. Batch-Suffixed Filenames**
```
profiles/
  geometry_batch12345.json  # Oct 28
  geometry_batch12401.json  # Nov 3 (different batch!)
  monarch_batch56789.json
  ...
```

### **4. Smart Skip Logic**
```
[15/36] Geometry
  URL: https://onyxcoffeelab.com/products/geometry
  ⊘ Skipped: Already have batch #12345
  
[16/36] Monarch
  URL: https://onyxcoffeelab.com/products/monarch
  ✓ Saved: monarch_batch56790.json
  📝 New batch (previous: 1 batches)
```

---

## 📊 **Example Output**

### **First Run (Oct 28):**
```
✓ Dataset directory: onyx_dataset_2025_10_28/
✓ ADDITIVE mode: Date-stamped, won't overwrite previous scrapes
✓ Batch history loaded: 0 products tracked
✓ No previous scrapes found - starting fresh

...scraping...

✓ New profiles: 28/36
⊘ Skipped (existing): 0
✗ Failed: 8
📈 New profile rate: 77.8%
```

### **Second Run (Nov 3):**
```
✓ Dataset directory: onyx_dataset_2025_11_03/
✓ ADDITIVE mode: Date-stamped, won't overwrite previous scrapes
✓ Batch history loaded: 28 products tracked
  Found 1 previous scrape(s)

...scraping...

✓ New profiles: 5/36
⊘ Skipped (existing): 23  ← Already have these batches!
✗ Failed: 8
📈 New profile rate: 13.9%

📊 Historical tracking:
   Total products tracked: 28
   Total batches in history: 28
   New batches this scrape: 5
```

**Only 5 new profiles!** Because 23 products still have the same batch as Oct 28.

### **Third Run (Nov 10):**
```
✓ Dataset directory: onyx_dataset_2025_11_10/
✓ Batch history loaded: 33 products tracked
  Found 2 previous scrape(s)

...scraping...

✓ New profiles: 12/36
⊘ Skipped (existing): 16
✗ Failed: 8

📊 Total dataset across all scrapes:
   • 45 total profiles collected  ← Cumulative!
   • 33 unique products
```

---

## 🗂️ **Directory Structure**

After multiple scrapes:
```
your_project/
├── onyx_dataset_2025_10_28/
│   ├── complete_dataset.json    (28 profiles)
│   ├── dataset_summary.csv
│   ├── profiles/
│   │   ├── geometry_batch12345.json
│   │   ├── monarch_batch56789.json
│   │   └── ...
│   └── README.md
│
├── onyx_dataset_2025_11_03/
│   ├── complete_dataset.json    (5 NEW profiles)
│   ├── dataset_summary.csv
│   ├── profiles/
│   │   ├── geometry_batch12401.json  ← New batch!
│   │   ├── tropical_weather_batch77777.json
│   │   └── ...
│   └── README.md
│
└── onyx_dataset_2025_11_10/
    ├── complete_dataset.json    (12 NEW profiles)
    ├── dataset_summary.csv
    ├── profiles/
    │   ├── geometry_batch12467.json  ← Another new batch!
    │   └── ...
    └── README.md

Total: 28 + 5 + 12 = 45 profiles across 3 scrapes!
```

---

## 🔍 **How Batch Checking Works**

### **Step 1: Load History**
```python
# On Nov 3, builder loads Oct 28 data:
batch_history = {
    'Geometry': [
        {'batch_number': '12345', 'roast_date': '2025-10-25', 'source_dir': 'onyx_dataset_2025_10_28'}
    ],
    'Monarch': [
        {'batch_number': '56789', 'roast_date': '2025-10-24', 'source_dir': 'onyx_dataset_2025_10_28'}
    ],
    ...
}
```

### **Step 2: Check Each Product**
```python
# Scraping Geometry on Nov 3
current_batch = '12345'  # From metadata extraction

# Check history
if 'Geometry' in batch_history:
    previous_batches = batch_history['Geometry']
    for prev in previous_batches:
        if prev['batch_number'] == '12345':
            # Already have this batch!
            skip = True
            break
```

### **Step 3: Save or Skip**
- **If new batch:** Save with batch suffix
- **If existing batch:** Skip, move to next product

---

## 💡 **Usage Tips**

### **Weekly Scraping Schedule:**
```bash
# Week 1 (Oct 28)
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
# Get: 28 profiles

# Week 2 (Nov 3) - Check for updates
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
# Get: 5 new profiles (if batches changed)

# Week 3 (Nov 10) - Final collection
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
# Get: 12 new profiles

# Total: 45 profiles across 3 weeks!
```

### **Merge All Scrapes (Optional):**
```python
import json
import glob

# Find all dataset directories
datasets = glob.glob('onyx_dataset_*/complete_dataset.json')

all_profiles = []
for dataset_file in datasets:
    with open(dataset_file, 'r') as f:
        data = json.load(f)
        all_profiles.extend(data['profiles'])

print(f"Total profiles: {len(all_profiles)}")

# Remove duplicates by batch number
unique_profiles = {}
for profile in all_profiles:
    name = profile['metadata']['product_name']
    batch = profile['metadata'].get('roast_info', {}).get('batch', 'unknown')
    key = f"{name}_{batch}"
    
    if key not in unique_profiles:
        unique_profiles[key] = profile

print(f"Unique profiles: {len(unique_profiles)}")

# Save merged dataset
merged_data = {
    'dataset_info': {
        'total_profiles': len(unique_profiles),
        'sources': [d.split('/')[0] for d in datasets]
    },
    'profiles': list(unique_profiles.values())
}

import os
os.makedirs('onyx_dataset_merged', exist_ok=True)
with open('onyx_dataset_merged/complete_dataset.json', 'w') as f:
    json.dump(merged_data, f, indent=2, default=str)

print("✓ Merged dataset saved!")
```

---

## ✅ **Ready to Use!**

### **No Configuration Needed:**
- ✅ Date stamping: Automatic
- ✅ Batch tracking: Automatic
- ✅ Skip duplicates: Automatic
- ✅ Safe to re-run: Always

### **Just Run It:**
```bash
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
```

**That's it!** Run the same command today, next week, next month - it handles everything automatically! 🎉

---

## 📈 **Expected Results**

### **Scenario 1: Fast Updates (2-3 day roasting cycle)**
```
Oct 28: 28 profiles
Nov 1:  18 new profiles (50% changed)
Nov 4:  20 new profiles (55% changed)
Nov 7:  22 new profiles (60% changed)
Total:  88 profiles
```

### **Scenario 2: Medium Updates (Weekly cycle)**
```
Oct 28: 28 profiles
Nov 4:  12 new profiles (33% changed)
Nov 11: 15 new profiles (40% changed)
Total:  55 profiles
```

### **Scenario 3: Slow Updates (Bi-weekly cycle)**
```
Oct 28: 28 profiles
Nov 10: 8 new profiles (22% changed)
Total:  36 profiles
```

**Any scenario is good!** Even 36 profiles is excellent for validation. 🎯

---

## 🎊 **Summary**

### **What Changed from v3.0:**
- ✅ **Automatic date-stamped directories** (no manual editing!)
- ✅ **Global batch history tracking** (across all scrapes)
- ✅ **Smart duplicate detection** (skips existing batches)
- ✅ **Batch-suffixed filenames** (geometry_batch12345.json)
- ✅ **Progress reporting** (shows skipped vs. new)

### **What You Do:**
```bash
# Today
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py

# Next week
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py

# Week after
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py
```

**Same command every time!** 🚀

---

**Perfect for longitudinal data collection without any manual work!** ☕🤖

Go ahead and run it - it won't overwrite anything! 🎉
