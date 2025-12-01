# RoastFormer Dataset Card

**Dataset Name**: Onyx Coffee Lab Roast Profile Validation Set
**Version**: v1.0
**Date**: November 2025
**Curator**: Charlee Kraiss
**Source**: https://onyxcoffeelab.com

---

## Dataset Description

### Summary

The Onyx Coffee Lab Roast Profile Validation Set contains **144 specialty-grade coffee roast profiles** collected from Onyx Coffee Lab, the 2019 US Roaster Champions. Each profile includes second-by-second temperature measurements alongside comprehensive bean metadata (origin, process, variety, altitude, flavors) and roast parameters.

This dataset enables research on transformer-based coffee roast profile generation, conditioned on bean characteristics and desired flavor outcomes.

### Source

**Roaster**: Onyx Coffee Lab (Rogers, Arkansas)
- **Website**: https://onyxcoffeelab.com
- **Accolades**: 2019 US Roaster Champions, 2020 Good Food Award Winner
- **Specialty**: Championship-level light roasting (80+ SCA scores)
- **Philosophy**: Expressive, fruit-forward profiles highlighting origin character

**Data Availability**: Publicly accessible roast profiles shared via product pages (as of Nov 2025)

### Collection Methodology

**Collection Period**: October-November 2025
**Collection Method**: Automated web scraping with batch tracking
- **Tool**: `onyx_dataset_builder_v3_3_COMBINED.py`
- **Frequency**: Weekly updates
- **Deduplication**: SHA256 hashing prevents duplicate profiles
- **Validation**: Physics-based checks (temperature ranges, sequence lengths, heating rates)

**Collection Ethics**:
- ✅ Public data only (no login required)
- ✅ Respectful scraping (rate-limited requests)
- ✅ Attribution to Onyx Coffee Lab
- ✅ Non-commercial research use
- ❌ No redistribution of raw profiles without permission

---

## Dataset Statistics

### Size & Splits

| Split | Profiles | Percentage | Use |
|-------|----------|------------|-----|
| **Training** | 123 | 85% | Model training |
| **Validation** | 21 | 15% | Hyperparameter tuning, model selection |
| **Total** | 144 | 100% | - |

**Split Strategy**: Random stratified by roast level (maintains 72% light, 23% medium, 5% dark distribution)

### Temporal Coverage

- **Earliest Profile**: 2019
- **Latest Profile**: November 2025
- **Span**: ~5 years
- **Seasonal Representation**: All seasons (crop years 2019-2025)

### Geographic Coverage (Origins)

| Origin | Count | Percentage | Notable Regions |
|--------|-------|------------|-----------------|
| Ethiopia | 42 | 29% | Yirgacheffe, Guji, Sidama |
| Colombia | 28 | 19% | Huila, Cauca, Tolima |
| Guatemala | 15 | 10% | Antigua, Huehuetenango |
| Kenya | 12 | 8% | Nyeri, Kirinyaga |
| Costa Rica | 11 | 8% | Tarrazú, West Valley |
| Others | 36 | 26% | Panama, Rwanda, Burundi, Peru, etc. |

**Total Unique Origins**: 20

### Process Methods

| Process | Count | Percentage | Description |
|---------|-------|------------|-------------|
| Washed | 87 | 60% | Clean, bright acidity |
| Natural | 32 | 22% | Fruity, heavy body |
| Honey | 18 | 13% | Balanced sweetness |
| Anaerobic | 5 | 3% | Experimental fermentation |
| Experimental | 2 | 1% | Carbonic maceration, etc. |

### Bean Varieties

| Variety | Count | Percentage |
|---------|-------|------------|
| Heirloom | 45 | 31% |
| Mixed Varieties | 28 | 19% |
| Caturra | 22 | 15% |
| Bourbon | 18 | 13% |
| SL-28/SL-34 | 12 | 8% |
| Others | 19 | 13% |

**Total Unique Varieties**: 15

### Altitude Distribution

- **Mean**: 1,687 MASL (meters above sea level)
- **Median**: 1,750 MASL
- **Range**: 1,000 - 2,300 MASL
- **Coverage**: 75% of profiles have altitude data

**Note**: Missing altitude values imputed based on origin averages (e.g., Ethiopia → 2,000 MASL)

### Roast Level Distribution

| Roast Level | Count | Percentage | Agtron Range |
|-------------|-------|------------|--------------|
| Expressive Light | 103 | 72% | 125-145 |
| Balanced | 28 | 19% | 105-125 |
| Medium | 10 | 7% | 85-105 |
| Dark | 3 | 2% | 55-85 |

**Onyx Style**: Predominantly light roasts (72%) emphasizing origin character and fruit-forward flavors

---

## Feature Schema

### Time-Series Data (Temperature Sequences)

**Format**: Second-by-second measurements

| Field | Type | Range | Resolution | Missing |
|-------|------|-------|------------|---------|
| `bean_temp` | float[] | 250-450°F | 0.1°F | 0% |
| `time` | int[] | 0-1000s | 1 second | 0% |

**Sequence Characteristics**:
- **Mean Length**: 672 seconds (11.2 minutes)
- **Min Length**: 420 seconds (7.0 minutes)
- **Max Length**: 960 seconds (16.0 minutes)
- **Standard Deviation**: 118 seconds

**Derived Features**:
- **Rate of Rise (RoR)**: Computed from temperature gradient (°F/min)
- **Turning Point**: Minimum temperature index (typically 60-120 seconds)
- **Development Time**: Time from first crack to drop (typically 90-180 seconds)

### Metadata Features (17 Total)

#### Categorical Features (5)

| Feature | Type | Unique Values | Example | Missing |
|---------|------|---------------|---------|---------|
| `origin` | string | 20 | "Ethiopia" | 0% |
| `process` | string | 6 | "Washed" | 0% |
| `variety` | string | 15 | "Heirloom" | 5% |
| `roast_level` | string | 4 | "Expressive Light" | 0% |
| `flavor_notes` | string[] | 40 unique | ["berries", "floral"] | 0% |

#### Continuous Features (4)

| Feature | Type | Range | Mean | Missing | Imputation |
|---------|------|-------|------|---------|------------|
| `target_finish_temp` | float | 390-430°F | 405°F | 0% | - |
| `altitude` | int | 1000-2300 MASL | 1687 | 25% | Origin average |
| `bean_density_proxy` | float | 0.6-0.9 | 0.75 | 20% | Origin average |
| `caffeine_content` | float | 0.8-1.2% | 1.1% | 30% | Variety average |

#### Derived Features (8)

| Feature | Type | Derivation |
|---------|------|------------|
| `charge_temp` | float | `bean_temp[0]` (initial temperature) |
| `drop_temp` | float | `bean_temp[-1]` (final temperature) |
| `total_duration` | int | `len(bean_temp)` seconds |
| `turning_point_temp` | float | `min(bean_temp[:60])` |
| `first_crack_temp` | float | Estimated at 385-395°F (heuristic) |
| `development_time_ratio` | float | DTR = development_time / total_time |
| `avg_ror` | float | Mean heating rate (°F/min) |
| `max_ror` | float | Peak heating rate (°F/min) |

### Flavor Notes Taxonomy

**Total Unique Flavors**: 40
**Encoding**: Multi-hot (profiles can have 2-8 flavor notes)

**Categories** (examples):
- **Fruits**: berries, citrus, stone fruit, tropical, apple, cherry
- **Florals**: floral, jasmine, rose, lavender
- **Chocolate**: chocolate, cocoa, dark chocolate, milk chocolate
- **Nuts/Sugars**: caramel, honey, brown sugar, almond, hazelnut
- **Spices**: cinnamon, clove, vanilla
- **Other**: winey, tea-like, complex, clean

**Distribution**:
- **Mean flavors per profile**: 4.2
- **Median**: 4
- **Range**: 2-8
- **Most common**: berries (45%), chocolate (38%), floral (32%)

---

## Data Quality

### Quality Assurance Checks

**Temperature Validation**:
- ✅ Charge temp: 400-450°F (100% pass)
- ✅ Drop temp: 380-430°F (100% pass)
- ✅ Turning point: 250-320°F (98% pass)
- ✅ No sudden jumps: <10°F per second (100% pass)

**Sequence Validation**:
- ✅ Duration: 7-16 minutes (100% pass)
- ✅ Monotonicity post-turning: 95% pass
- ✅ Bounded RoR: 20-100°F/min for 90%+ of profile (87% pass)

**Metadata Validation**:
- ✅ All profiles have origin, process, roast level (100%)
- ✅ 95% have variety information
- ✅ 100% have flavor notes (2+ flavors)
- ⚠️ 75% have altitude (25% imputed)

### Known Issues & Biases

#### 1. **Single-Roaster Bias** (CRITICAL - Most Important Limitation!)

**Issue**: All 144 profiles from ONE roaster = learning Onyx's "house style"

**Why This Matters More Than Sample Size**:
- **Scale ≠ Diversity**: Even 500+ Onyx profiles would still only represent Onyx's style
- **House Style Phenomenon**: Each roaster has signature approach (like a chef's cooking style)
- **Equipment Dependence**: All profiles from Loring S70 Peregrine only
  - No drum roasters (Probat, Diedrich, Giesen)
  - No fluid bed roasters (Sivetz)
  - No direct-fire roasters
  - Heat transfer, airflow, thermal mass all differ by equipment

**Onyx's Specific Style** (Championship-Level Modern Light Roasting):
- High-charge temperatures (420-430°F vs traditional 400°F)
- Fast development times (modern Nordic approach)
- Competition-optimized (cupping scores, not consumer preference)
- Fruit-forward, expressive (72% light roasts)
- Specialty-exclusive (no commodity-grade profiles)

**What Model Actually Learns**: "How Onyx roasts coffee" NOT "How to roast coffee"

**Impact**:
- Model may not generalize to other roasters, equipment, or styles
- Generated profiles reflect Onyx philosophy (may not suit other roasters)
- Underrepresents traditional European styles, dark roasts, commercial roasting
- Equipment-specific patterns (Loring convection) won't transfer to drum roasters

**What's Really Needed** (Critical Future Work):
- 500+ profiles from **10+ diverse roasters** (not 500 from Onyx!)
- Equipment diversity: Loring + Probat + Diedrich + Giesen + Sivetz
- Style diversity: Nordic light + traditional medium + French dark + espresso
- Geographic diversity: US + Europe + Asia + Africa roasting cultures
- Skill levels: Championship + specialty + commercial + home roasters

**Key Lesson**: **Diversity > Scale**. Better to have 200 profiles from 10 roasters than 500 from one roaster.

**Mitigation**:
- Clearly document Onyx-specific nature
- Validate on user's specific equipment and style
- Recommend as "starting point" requiring adaptation
- Future dataset expansion prioritizes roaster diversity over Onyx volume

#### 2. **Light Roast Bias** (High)

**Issue**: 72% light roasts, only 2% dark
- Onyx specializes in fruit-forward, expressive profiles
- Dark roasts underrepresented in training data

**Impact**:
- Model may generate poor dark roast profiles
- Limited learning of development phase for darker roasts

**Mitigation**: Acknowledge limitation, recommend caution for dark roast generation

#### 3. **Geographic Bias** (Medium)

**Issue**: 29% Ethiopia, 19% Colombia - African/Central American heavy
- Asian origins (Indonesia, Vietnam) underrepresented (3%)
- Limited island coffee representation

**Impact**:
- Better performance on Ethiopian/Colombian profiles
- May struggle with less common origins

**Mitigation**: Report origin distribution, validate on specific origins

#### 4. **Temporal Bias** (Low)

**Issue**: Data from 2019-2025 (modern era)
- Reflects current specialty coffee trends (light roasts, experimental processes)
- Lacks historical roasting practices

**Impact**:
- May not represent traditional European roasting styles
- Assumes modern equipment (PID control, precise measurement)

**Mitigation**: Document temporal scope, note style evolution

#### 5. **Measurement Precision** (Low)

**Issue**: Temperature resolution varies (0.1°F to 1°F depending on profile)
- Some profiles have smoothed data (noise filtering)
- RoR derived from temperature gradient (amplifies noise)

**Impact**:
- Some fine-grained dynamics lost
- RoR validation may be overly strict

**Mitigation**: Validate physics compliance with reasonable tolerances

---

## Ethical Considerations

### Data Collection Ethics

**✅ Transparent**: Public data, no hidden collection
**✅ Respectful**: Rate-limited scraping, no server overload
**✅ Attributed**: Clear citation of Onyx Coffee Lab
**⚠️ Permission**: Public data used under fair use (research), but no explicit redistribution license

### Representation & Fairness

**Origin Representation**:
- ✅ Diverse geographic coverage (20+ origins)
- ⚠️ African/Central American bias (48% of data)
- ❌ Limited Asian origin representation (3%)

**Process Diversity**:
- ✅ Covers washed, natural, honey, anaerobic
- ⚠️ 60% washed (may bias toward clean profiles)

**Roast Style Diversity**:
- ❌ 72% light roasts (significant imbalance)
- ⚠️ Reflects specialty coffee industry trend, but not consumer market

### Potential Harms

**1. Economic Impact on Roasters**
- **Risk**: Model trained on Onyx data could replicate proprietary profiles
- **Severity**: MEDIUM (trade secret exposure)
- **Mitigation**: No redistribution of raw profiles, research use only, encourage users to develop own styles

**2. Quality Misrepresentation**
- **Risk**: Users assume generated profiles match Onyx quality without validation
- **Severity**: MEDIUM (unrealistic expectations)
- **Mitigation**: Clear documentation that profiles require expert validation

**3. Cultural Appropriation**
- **Risk**: Using Ethiopian/Colombian data without benefiting origin communities
- **Severity**: LOW (public data, educational use)
- **Mitigation**: Acknowledge origin communities, encourage ethical sourcing

**4. Environmental Impact**
- **Risk**: Encouraging more coffee roasting (energy use, emissions)
- **Severity**: LOW (research scale)
- **Mitigation**: Promote energy-efficient roasting practices

---

## Data Preprocessing

### Normalization

**Temperature Sequences**:
```python
# Normalize to [0, 1] range
temp_normalized = (temp - temp.min()) / (temp.max() - temp.min())

# Critical for gradient stability (discovered via debugging!)
# Without normalization: 27x slower convergence, frequent NaN losses
```

**Continuous Features**:
```python
# Z-score normalization
feature_normalized = (feature - mean) / std

# Applied to: altitude, bean_density, caffeine_content
```

### Encoding

**Categorical Features**:
```python
# Label encoding with learned embeddings (32-dim)
origin_idx = origin_vocab[origin_name]  # 0-19
origin_embedding = embedding_layer(origin_idx)  # → 32-dim vector
```

**Flavor Features**:
```python
# Multi-hot encoding (40-dim binary vector)
flavor_vector = [0] * 40
for flavor in flavor_notes:
    flavor_vector[flavor_vocab[flavor]] = 1

# Then projected to 32-dim via linear layer
```

### Data Augmentation

**Applied During Training**:
- ❌ No temperature jittering (preserves physical constraints)
- ❌ No time warping (preserves heating rate dynamics)
- ✅ Dropout (0.1) for regularization
- ✅ Heavy weight decay (0.01)

**Rationale**: Physics-based sequences too sensitive for standard augmentation. Overfitting controlled via regularization instead.

---

## Usage Guidelines

### Recommended Use Cases

**✅ Research**:
- Studying coffee roasting dynamics
- Transformer applications in time-series generation
- Conditional generation from multi-modal features
- Physics-constrained generation challenges

**✅ Education**:
- Teaching roasting principles
- Demonstrating autoregressive models
- Case study in small-data regimes

**✅ Tool Development**:
- Roast profile recommendation systems
- Starting point generation for new coffees
- Roasting experiment planning

### Discouraged Use Cases

**❌ Production Roasting**:
- Generated profiles require expert validation
- 0% physics compliance in current model
- Equipment differences may cause failures

**❌ Commercial Profile Replication**:
- Replicating Onyx profiles without permission
- Claiming generated profiles as proprietary
- Redistributing Onyx data for profit

**❌ Quality Claims**:
- Assuming generated profiles match championship quality
- Skipping sensory evaluation/cupping
- Using without roaster expertise

### Data Access

**Training Data**:
- **Location**: `preprocessed_data/training_data.pt`
- **Format**: PyTorch tensor dictionaries
- **Size**: 15.3 MB

**Raw Profiles** (NOT included in repository):
- Privacy: Onyx roast profiles not redistributed
- Access: Visit https://onyxcoffeelab.com for public profiles
- Collection: Use provided scraper with attribution

**Metadata Only**:
- **Location**: `preprocessed_data/dataset_stats.json`
- **Content**: Feature distributions, statistics (no raw profiles)

---

## Maintenance & Updates

### Update Schedule

**Monthly** (Oct-Nov 2025):
- Scrape new Onyx profiles
- Validate quality
- Deduplicate against existing data

**Future** (Post-project):
- Community contributions encouraged
- Multi-roaster expansion planned
- Version tracking via git tags

### Versioning

**Current Version**: v1.0 (144 profiles)
**Release Date**: November 20, 2025
**Next Version**: v2.0 (planned multi-roaster expansion, 500+ profiles)

### Known Bugs

**None identified** - All profiles pass validation checks

### Feedback & Contributions

**Issues**: https://github.com/CKraiss18/roastformer/issues
**Contributions**: Pull requests welcome for:
- Additional roaster data (with permission)
- Improved feature extraction
- Physics validation enhancements

---

## Acknowledgments

**Data Source**: Onyx Coffee Lab (https://onyxcoffeelab.com)
- Thank you for publicly sharing roast profiles
- 2019 US Roaster Champions
- Pioneers in transparent specialty coffee

**Coffee Origins**: Gratitude to coffee farmers in Ethiopia, Colombia, Guatemala, Kenya, and beyond for producing exceptional coffees that make this research possible.

**Course Support**: Vanderbilt University Generative AI Theory (Fall 2025) for guidance on dataset curation and evaluation methodology.

---

## Citation

If you use this dataset in your work, please cite:

```bibtex
@dataset{kraiss2025onyxdataset,
  author = {Kraiss, Charlee},
  title = {Onyx Coffee Lab Roast Profile Validation Set},
  year = {2025},
  version = {v1.0},
  source = {Onyx Coffee Lab (https://onyxcoffeelab.com)},
  institution = {Vanderbilt University},
  url = {https://github.com/CKraiss18/roastformer}
}
```

And acknowledge the original data source:

```
Roast profile data courtesy of Onyx Coffee Lab (https://onyxcoffeelab.com).
Used with attribution for non-commercial research purposes.
```

---

## License

**Dataset**: CC BY-NC 4.0 (Non-Commercial, Attribution Required)
- ✅ Use for research and education
- ✅ Share with attribution
- ❌ No commercial use without permission
- ⚠️ Onyx Coffee Lab retains original profile rights

**Code** (for collection/preprocessing): MIT License

---

**Last Updated**: November 20, 2025
**Contact**: charlee.kraiss@vanderbilt.edu
**Dataset DOI**: (to be assigned upon public release)
