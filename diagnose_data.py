"""
Diagnostic: Check Flavor and Process Data
Run this to see what's in your dataset
"""

import pandas as pd
import glob
import json
from pathlib import Path

print("="*60)
print("FLAVOR & PROCESS DIAGNOSTIC")
print("="*60)

# Find all datasets
csv_files = glob.glob('onyx_dataset_*/dataset_summary.csv')

if not csv_files:
    print("❌ No dataset CSV files found!")
    exit()

print(f"\n✅ Found {len(csv_files)} dataset(s)\n")

for csv_file in csv_files:
    print(f"📂 {Path(csv_file).parent.name}")
    print("-"*60)
    
    df = pd.read_csv(csv_file)
    
    # Check columns
    print("\n1. COLUMNS AVAILABLE:")
    flavor_cols = [col for col in df.columns if 'flavor' in col.lower()]
    process_cols = [col for col in df.columns if 'process' in col.lower()]
    
    print(f"   Flavor columns: {flavor_cols}")
    print(f"   Process columns: {process_cols}")
    
    # Check process values
    print("\n2. PROCESS VALUES:")
    if 'process' in df.columns:
        print(f"   Unique processes: {df['process'].nunique()}")
        print(f"   Values: {df['process'].dropna().unique().tolist()}")
        print(f"   Missing: {df['process'].isna().sum()} / {len(df)}")
    else:
        print("   ❌ No 'process' column!")
    
    # Check flavor data
    print("\n3. FLAVOR DATA:")
    if 'flavor_notes_raw' in df.columns:
        print(f"   Raw flavors present: {df['flavor_notes_raw'].notna().sum()} / {len(df)}")
        print(f"   Example: {df['flavor_notes_raw'].dropna().iloc[0] if df['flavor_notes_raw'].notna().any() else 'None'}")
    else:
        print("   ❌ No 'flavor_notes_raw' column!")
    
    if 'flavor_notes_parsed' in df.columns:
        print(f"   Parsed flavors present: {df['flavor_notes_parsed'].notna().sum()} / {len(df)}")
        if df['flavor_notes_parsed'].notna().any():
            example = df['flavor_notes_parsed'].dropna().iloc[0]
            print(f"   Example: {example}")
            print(f"   Type: {type(example)}")
    else:
        print("   ❌ No 'flavor_notes_parsed' column!")
    
    # Sample a few products
    print("\n4. SAMPLE PRODUCTS (First 3):")
    for idx, row in df.head(3).iterrows():
        print(f"\n   Product: {row['product_name']}")
        print(f"   Process: {row.get('process', 'MISSING')}")
        print(f"   Flavor Raw: {row.get('flavor_notes_raw', 'MISSING')}")
        print(f"   Flavor Parsed: {row.get('flavor_notes_parsed', 'MISSING')}")
    
    print("\n" + "="*60 + "\n")

# Now check a JSON profile file
print("\n📄 CHECKING JSON PROFILE FILES:")
print("-"*60)

json_files = list(Path('onyx_dataset_2024_11_03/profiles').glob('*.json')) if Path('onyx_dataset_2024_11_03/profiles').exists() else []

if not json_files:
    # Try other dataset directories
    for dataset_dir in glob.glob('onyx_dataset_*'):
        json_files = list(Path(dataset_dir).glob('profiles/*.json'))
        if json_files:
            break

if json_files:
    print(f"✅ Found {len(json_files)} JSON files")
    
    # Load one example
    with open(json_files[0], 'r') as f:
        profile = json.load(f)
    
    print(f"\n5. EXAMPLE JSON STRUCTURE:")
    print(f"   File: {json_files[0].name}")
    
    if 'metadata' in profile:
        metadata = profile['metadata']
        print(f"\n   Metadata keys: {list(metadata.keys())[:10]}...")
        
        print(f"\n   Process: {metadata.get('process', 'MISSING')}")
        print(f"   Flavor Raw: {metadata.get('flavor_notes_raw', 'MISSING')}")
        print(f"   Flavor Parsed: {metadata.get('flavor_notes_parsed', 'MISSING')}")
        print(f"   Flavor Categories: {metadata.get('flavor_categories', 'MISSING')}")
else:
    print("❌ No JSON profile files found!")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\n📋 FINDINGS SUMMARY:")
print("   Copy the output above and share with Claude")
print("   We'll fix any issues found!")
