# Enhanced Dataset Builder v3.0 - Feature Extraction Guide

## 🎯 What's New in V3.0

**Phase 1 + Phase 2 Features** now automatically extracted for transformer conditioning!

### Phase 1: Critical Features (Always Needed)
1. ✅ **Origin** - Colombia, Ethiopia, Kenya, etc.
2. ✅ **Process** - Washed, Natural, Honey, Anaerobic
3. ✅ **Roast Level** - Expressive Light, Medium, Dark
4. ✅ **Roast Level Agtron** - #135 numeric value
5. ✅ **Target Finish Temp** - Inferred from Agtron (395-425°F)

### Phase 2: Helpful Features (Improve Quality)
6. ✅ **Variety** - Mixed, Heirloom, Caturra, Bourbon, Geisha
7. ✅ **Altitude** - 1500 MASL, 1800-2200m
8. ✅ **Altitude Numeric** - Average altitude in meters
9. ✅ **Bean Density Proxy** - Calculated from altitude (g/cm³)
10. ✅ **Drying Method** - Raised-Bed, Patio, African Bed, Mechanical

---

## 📊 Expected Feature Coverage

Based on Onyx's product pages:

| Feature | Coverage | Example Values |
|---------|----------|----------------|
| Origin | 95-100% | Colombia, Ethiopia, Kenya, Costa Rica |
| Process | 80-90% | Washed, Natural, Honey, Anaerobic |
| Roast Level | 100% | Expressive Light, Medium, Dark |
| Agtron | 100% | #135, #95, #75 |
| Target Finish Temp | 100% | 395-425°F (inferred) |
| Variety | 70-90% | Mixed, Heirloom, Caturra, Bourbon, Geisha |
| Altitude | 60-80% | 1500 MASL, 1800-2200m |
| Bean Density | 60-80% | 0.70-0.78 g/cm³ (calculated) |
| Drying Method | 40-60% | Raised-Bed, Patio, African Bed |
| Harvest Season | 60-80% | Rotating Microlots, October, 2024 |
| Roaster Machine | 90-100% | Loring S70 Peregrine |
| Extraction | 100% | Filter & Espresso, Filter, Espresso |
| Caffeine | 100% | 215mg, 195mg |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install selenium pandas beautifulsoup4

# Run enhanced builder
python onyx_dataset_builder_v3_enhanced.py

# Expected output: 25-35 profiles with full feature extraction
```

---

## 📈 What You'll Get

### CSV Output (dataset_summary.csv)
```csv
product_name,origin,process,roast_level,roast_level_agtron,target_finish_temp,variety,altitude,altitude_numeric,bean_density_proxy,drying_method,...
Geometry,Colombia & Ethiopia,Washed,Expressive Light,135,395.0,Mixed,1500 MASL,1500,0.725,Raised-Bed,...
Monarch,Ethiopia,Natural,Light,125,395.0,Heirloom,2000 MASL,2000,0.75,Patio,...
...
```

### JSON Profile Structure
```json
{
  "metadata": {
    "product_name": "Geometry",
    "origin": "Colombia, Ethiopia",
    "process": "Washed",
    "roast_level": "Expressive Light",
    "roast_level_agtron": 135,
    "target_finish_temp": 395.0,
    "variety": "Mixed",
    "altitude": "1500 MASL",
    "altitude_numeric": 1500,
    "bean_density_proxy": 0.725,
    "drying_method": "Raised-Bed Dried",
    "harvest_season": "Rotating Microlots",
    "roaster_machine": "Loring S70 Peregrine",
    "preferred_extraction": "Filter & Espresso",
    "caffeine_mg": 215
  },
  "roast_profile": {
    "bean_temp": [...],
    "rate_of_rise": [...]
  },
  "summary": {...}
}
```

---

## 🤖 Using Features for Transformer Conditioning

### Step 1: Load Dataset
```python
import pandas as pd
import json

# Load CSV summary
df = pd.read_csv('onyx_dataset/dataset_summary.csv')

# Check feature coverage
print("Feature Coverage:")
for col in ['origin', 'process', 'roast_level', 'variety', 'altitude', 'drying_method']:
    coverage = (df[col].notna().sum() / len(df)) * 100
    print(f"  {col}: {coverage:.1f}%")
```

### Step 2: Create Feature Encoders
```python
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Categorical features
origin_encoder = LabelEncoder()
process_encoder = LabelEncoder()
variety_encoder = LabelEncoder()
roast_level_encoder = LabelEncoder()

# Fit encoders
origins = df['origin'].dropna().unique()
processes = df['process'].dropna().unique()
varieties = df['variety'].dropna().unique()
roast_levels = df['roast_level'].dropna().unique()

origin_encoder.fit(origins)
process_encoder.fit(processes)
variety_encoder.fit(varieties)
roast_level_encoder.fit(roast_levels)

print(f"Unique origins: {len(origins)} - {list(origins)[:5]}...")
print(f"Unique processes: {len(processes)} - {list(processes)}")
print(f"Unique varieties: {len(varieties)} - {list(varieties)[:5]}...")
print(f"Unique roast levels: {len(roast_levels)} - {list(roast_levels)}")
```

### Step 3: Create Embeddings
```python
# Embedding dimensions
embed_dim = 32

# Create embedding layers
origin_embed = nn.Embedding(len(origins), embed_dim)
process_embed = nn.Embedding(len(processes), embed_dim)
variety_embed = nn.Embedding(len(varieties), embed_dim)
roast_level_embed = nn.Embedding(len(roast_levels), embed_dim)

# Continuous feature projection
continuous_dim = 4  # finish_temp, altitude, density, caffeine
continuous_proj = nn.Linear(continuous_dim, embed_dim)
```

### Step 4: Prepare Conditioning Vector
```python
def prepare_condition(row):
    """Prepare conditioning vector for a single profile"""
    
    # Categorical features (get indices)
    origin_idx = origin_encoder.transform([row['origin']])[0] if pd.notna(row['origin']) else 0
    process_idx = process_encoder.transform([row['process']])[0] if pd.notna(row['process']) else 0
    variety_idx = variety_encoder.transform([row['variety']])[0] if pd.notna(row['variety']) else 0
    roast_idx = roast_level_encoder.transform([row['roast_level']])[0] if pd.notna(row['roast_level']) else 0
    
    # Continuous features (normalized)
    finish_temp_norm = row['target_finish_temp'] / 425.0  # Normalize to ~0-1
    altitude_norm = row['altitude_numeric'] / 2500.0 if pd.notna(row['altitude_numeric']) else 0.6
    density_norm = row['bean_density_proxy'] / 0.80 if pd.notna(row['bean_density_proxy']) else 0.85
    caffeine_norm = row['caffeine_mg'] / 230.0 if pd.notna(row['caffeine_mg']) else 0.9
    
    continuous_features = torch.tensor([
        finish_temp_norm,
        altitude_norm,
        density_norm,
        caffeine_norm
    ], dtype=torch.float32)
    
    # Get embeddings
    origin_emb = origin_embed(torch.tensor(origin_idx))
    process_emb = process_embed(torch.tensor(process_idx))
    variety_emb = variety_embed(torch.tensor(variety_idx))
    roast_emb = roast_level_embed(torch.tensor(roast_idx))
    continuous_emb = continuous_proj(continuous_features)
    
    # Concatenate all
    condition_vector = torch.cat([
        origin_emb,
        process_emb,
        variety_emb,
        roast_emb,
        continuous_emb
    ])
    
    return condition_vector  # Shape: (embed_dim * 5,)

# Example usage
sample_row = df.iloc[0]
condition = prepare_condition(sample_row)
print(f"Condition vector shape: {condition.shape}")  # (160,) if embed_dim=32
```

### Step 5: Use in Transformer
```python
class RoastFormer(nn.Module):
    def __init__(self, condition_dim=160, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # Project condition to model dimension
        self.condition_proj = nn.Linear(condition_dim, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)  # Predict temperature
    
    def forward(self, tgt_sequence, condition):
        # Project condition
        condition_embed = self.condition_proj(condition)  # (batch, d_model)
        
        # Add condition to each timestep (broadcast)
        tgt_embed = tgt_sequence + condition_embed.unsqueeze(1)
        
        # Run transformer
        output = self.transformer(tgt_embed, memory=condition_embed.unsqueeze(0))
        
        # Project to temperature
        temps = self.output_proj(output)
        
        return temps

# Usage
model = RoastFormer()
condition = prepare_condition(sample_row)
# ... generate temperature sequence
```

---

## 🎯 Feature Importance for RoastFormer

### Critical (Always Use)
1. **Target Finish Temp** - Direct target for generation
2. **Roast Level / Agtron** - Determines development strategy
3. **Origin** - Affects bean characteristics, roasting style

### Very Helpful
4. **Process** - Affects moisture, density, structure
5. **Altitude / Density** - Directly impacts heating rate
6. **Variety** - Correlates with size, density, optimal profile

### Nice to Have
7. **Drying Method** - Additional density/moisture signal
8. **Roaster Machine** - Machine-specific thermal characteristics
9. **Extraction Method** - Target use case (filter vs espresso)

---

## 📝 Validation Tips

### Check Feature Quality
```python
# After scraping, verify feature extraction quality

# 1. Check for missing values
print("Missing values:")
print(df[['origin', 'process', 'variety', 'altitude', 'drying_method']].isna().sum())

# 2. Check value distributions
print("\nOrigin distribution:")
print(df['origin'].value_counts())

print("\nProcess distribution:")
print(df['process'].value_counts())

print("\nRoast level distribution:")
print(df['roast_level'].value_counts())

# 3. Check altitude range
print(f"\nAltitude range: {df['altitude_numeric'].min():.0f}-{df['altitude_numeric'].max():.0f}m")

# 4. Check finish temp range
print(f"Finish temp range: {df['target_finish_temp'].min():.0f}-{df['target_finish_temp'].max():.0f}°F")
```

### Handle Missing Values
```python
# Strategy 1: Use defaults for missing values
df['altitude_numeric'].fillna(1500, inplace=True)  # Default mid-altitude
df['bean_density_proxy'].fillna(0.72, inplace=True)  # Default mid-density

# Strategy 2: Use origin-based defaults
altitude_by_origin = {
    'Ethiopia': 2000,
    'Colombia': 1600,
    'Kenya': 1800,
    'Costa Rica': 1400,
}

for origin, alt in altitude_by_origin.items():
    mask = (df['origin'].str.contains(origin, na=False)) & (df['altitude_numeric'].isna())
    df.loc[mask, 'altitude_numeric'] = alt

# Strategy 3: Drop profiles with too many missing features
required_features = ['origin', 'process', 'roast_level']
df_clean = df.dropna(subset=required_features)
print(f"Profiles after cleaning: {len(df_clean)}/{len(df)}")
```

---

## 🔥 Expected Results

After running the enhanced builder:

```
✓ Discovered 36 products
✓ Successfully scraped 28-32 profiles (78-89% success rate)
✓ Phase 1 features: 100% coverage (origin, process, roast level)
✓ Phase 2 features: 60-80% coverage (variety, altitude, drying)
✓ Ready for transformer training!
```

**Feature Coverage Summary:**
- High coverage (>90%): Origin, Process, Roast Level, Finish Temp
- Medium coverage (60-90%): Variety, Altitude, Harvest Season
- Lower coverage (40-60%): Drying Method
- Always present: Roaster Machine, Extraction, Caffeine

This gives you a **robust conditioning framework** for RoastFormer! 🎯☕🤖

---

## 📞 Next Steps

1. **Run the enhanced builder**: `python onyx_dataset_builder_v3_enhanced.py`
2. **Check feature coverage**: Review CSV for quality
3. **Design conditioning architecture**: Decide on embedding sizes
4. **Implement feature encoders**: Build the conditioning pipeline
5. **Start training**: Use features to condition RoastFormer

Perfect timing for your Nov 3-8 implementation phase! 🚀
