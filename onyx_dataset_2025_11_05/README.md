# Onyx Coffee Lab Roast Profile Dataset - ENHANCED v3.0

**Created:** 2025-11-05 21:34  
**Source:** https://onyxcoffeelab.com  
**Purpose:** Validation dataset for RoastFormer with comprehensive bean characteristics  
**Features:** Phase 1 + Phase 2 conditioning variables

## Dataset Overview

- **Total Profiles:** 24
- **Success Rate:** 48.0%
- **Version:** v3.1_additive

## Enhanced Features

### Phase 1: Critical Conditioning Variables
1. **Origin** - Geographic region (Ethiopia, Colombia, Kenya, etc.)
2. **Process** - Processing method (Washed, Natural, Honey, Anaerobic)
3. **Roast Level** - Target roast description (Expressive Light, Medium, Dark)
4. **Roast Level Agtron** - Numeric Agtron value (e.g., #135)
5. **Target Finish Temp** - Inferred target finish temperature (°F)

### Phase 2: Helpful Features
6. **Variety** - Coffee variety (Mixed, Heirloom, Caturra, Bourbon, Geisha)
7. **Altitude** - Growing altitude (e.g., "1500 MASL")
8. **Altitude Numeric** - Numeric altitude in meters
9. **Bean Density Proxy** - Calculated from altitude (g/cm³)
10. **Drying Method** - Post-harvest drying (Raised-Bed, Patio, etc.)

### Phase 3: Flavor Profile Features (NEW!)
11. **Flavor Notes Raw** - Raw flavor text ("BERRIES STONE FRUIT EARL GREY HONEYSUCKLE ROUND")
12. **Flavor Notes Parsed** - Individual flavor descriptors as list
13. **Flavor Categories** - Categorized flavors (fruity, floral, chocolate, etc.)

### Additional Context
11. **Harvest Season** - When harvested (Rotating Microlots, October, etc.)
12. **Roaster Machine** - Production roaster (Loring S70 Peregrine, etc.)
13. **Preferred Extraction** - Intended use (Filter, Espresso, Both)
14. **Caffeine Content** - Caffeine in mg per 12oz cup

## Transformer Conditioning Usage

### Categorical Features (Embedding Layers)
```python
# Origin: {'Ethiopia': 0, 'Colombia': 1, 'Kenya': 2, ...}
# Process: {'Washed': 0, 'Natural': 1, 'Honey': 2, 'Anaerobic': 3}
# Roast Level: {'Light': 0, 'Medium': 1, 'Dark': 2}
# Variety: {'Mixed': 0, 'Heirloom': 1, 'Caturra': 2, ...}
```

### Continuous Features (Direct Input)
```python
# target_finish_temp: 395-425°F (normalized 0-1)
# altitude_numeric: 1000-2500m (normalized 0-1)
# bean_density_proxy: 0.65-0.80 g/cm³ (normalized 0-1)
# caffeine_mg: 180-230mg (normalized 0-1)
```

### Example Conditioning Code
```python
import torch
import torch.nn as nn

# Embedding layers for categorical features
origin_embed = nn.Embedding(num_origins, embed_dim)
process_embed = nn.Embedding(num_processes, embed_dim)
roast_level_embed = nn.Embedding(num_roast_levels, embed_dim)
variety_embed = nn.Embedding(num_varieties, embed_dim)

# Flavor embeddings (average multiple flavor notes)
flavor_embed = nn.Embedding(num_flavors, embed_dim)

# Continuous feature projection
continuous_features = torch.tensor([
    target_finish_temp / 425.0,  # Normalize to 0-1
    altitude_numeric / 2500.0,
    bean_density_proxy / 0.80,
])
continuous_proj = nn.Linear(3, embed_dim)(continuous_features)

# Flavor encoding (average embeddings for multiple flavors)
flavor_indices = [flavor_vocab['berries'], flavor_vocab['floral'], flavor_vocab['citrus']]
flavor_embeds = [flavor_embed(torch.tensor(idx)) for idx in flavor_indices]
avg_flavor_embed = torch.mean(torch.stack(flavor_embeds), dim=0)

# Combine all conditioning
condition_vector = torch.cat([
    origin_embed(origin_idx),
    process_embed(process_idx),
    roast_level_embed(roast_level_idx),
    variety_embed(variety_idx),
    avg_flavor_embed,  # NEW: Flavor conditioning!
    continuous_proj
], dim=-1)

# Feed to transformer
output = transformer(temperature_sequence, condition=condition_vector)
```

## Directory Structure

```
onyx_dataset_2025_11_05/
├── complete_dataset.json      # Full dataset with all features
├── dataset_summary.csv         # CSV with all Phase 1 + Phase 2 features
├── product_urls.json           # Discovered product URLs
├── profiles/                   # Individual profile JSONs
├── logs/                       # Progress checkpoints
└── README.md                   # This file
```

## Feature Coverage

Expected coverage rates:
- **Origin:** ~95-100% (almost always present)
- **Process:** ~80-90% (common but not always listed)
- **Roast Level:** ~100% (always present)
- **Variety:** ~70-90% (varies by coffee)
- **Altitude:** ~60-80% (not always specified)
- **Drying Method:** ~40-60% (less commonly detailed)

## Bean Density Calculation

Bean density proxy is estimated from altitude:
```python
# Formula: density = 0.65 + (altitude_km * 0.05)
# Example: 2000m altitude → 0.65 + (2.0 * 0.05) = 0.75 g/cm³
# 
# Typical ranges:
# - Low altitude (< 1000m): 0.65-0.70 g/cm³
# - Mid altitude (1000-1800m): 0.70-0.74 g/cm³
# - High altitude (1800-2500m): 0.74-0.78 g/cm³
```

## Roast Level to Temperature Mapping

Target finish temperatures inferred from Agtron values:
- **Agtron 120+** (Very Light): 395°F
- **Agtron 100-119** (Light): 405°F
- **Agtron 80-99** (Medium-Light): 410°F
- **Agtron 60-79** (Medium): 415°F
- **Agtron 50-59** (Medium-Dark): 420°F
- **Agtron < 50** (Dark): 425°F

## Citation

```
Onyx Coffee Lab Enhanced Roast Profile Dataset v3.0
Source: https://onyxcoffeelab.com
Accessed: November 2025
Features: Phase 1 + Phase 2 conditioning variables for transformer training
Purpose: RoastFormer validation dataset
```

## Version History

- **v3.0** (2025-11-05): Enhanced feature extraction
  - Added Phase 1 features: Origin, Process, Roast Level, Target Finish Temp
  - Added Phase 2 features: Variety, Altitude, Bean Density, Drying Method
  - Added contextual features: Harvest Season, Roaster Machine, Extraction Method
  - 24 roast profiles

---

**Ready for RoastFormer Transformer Training! ☕🤖**
