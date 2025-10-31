# ☕ RoastFormer: Transformer-Based Coffee Roast Profile Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

> A novel deep learning approach to generating coffee roast profiles using transformer architecture with multi-modal conditioning on bean characteristics and target flavor profiles.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

RoastFormer is a transformer-based model that generates coffee roast profiles conditioned on:
- **Bean characteristics** (origin, process, variety, altitude)
- **Target roast level** (light, medium, dark)
- **Desired flavor profile** (fruity, chocolate, floral, etc.)

### **The Problem**
Traditional coffee roasting requires years of experience to develop recipes that achieve specific flavor outcomes. Roasters must manually experiment with temperature curves, often taking weeks to perfect a single origin.

### **Our Solution**
RoastFormer learns the relationship between roast profiles and flavor outcomes from real specialty coffee data, enabling:
- **Flavor-guided generation**: "Generate a profile for bright, fruity notes"
- **What-if exploration**: "How would Ethiopian beans roast at 395°F finish?"
- **Recipe acceleration**: Rapid prototyping of new roast profiles

### **Novel Contributions**
1. **Flavor-conditioned generation** - First model to generate roast profiles from target flavor descriptors
2. **Real specialty data** - Validation dataset from Onyx Coffee Lab (award-winning roaster)
3. **Multi-modal conditioning** - Combines categorical, continuous, and semantic (flavor) features
4. **Longitudinal collection** - Tracks batch-to-batch consistency over time

---

## ✨ Key Features

### **Model Capabilities**
- ✅ Decoder-only transformer architecture (auto-regressive generation)
- ✅ Multi-modal conditioning (17 features across categorical, continuous, and semantic spaces)
- ✅ Flavor embedding layer for semantic control
- ✅ Physics-informed constraints (monotonic post-turning-point, bounded heating rates)
- ✅ Supports 3 positional encoding variants (sinusoidal, learned, RoPE)

### **Dataset Tools**
- ✅ Automated web scraper for Onyx Coffee Lab profiles
- ✅ Additive collection with batch tracking (no duplicates)
- ✅ Date-stamped directories for longitudinal analysis
- ✅ Comprehensive feature extraction (Phase 1 + Phase 2 + Flavor)
- ✅ CSV export for easy analysis

### **Validation Metrics**
- ✅ Mean Absolute Error (MAE) for temperature predictions
- ✅ Dynamic Time Warping (DTW) for curve similarity
- ✅ Monotonicity checks (post-turning-point)
- ✅ Heating rate bounds (20-100°F/min)
- ✅ Target finish temperature accuracy

---

## 📊 Dataset

### **Onyx Coffee Lab Validation Set**
Real roast profiles from [Onyx Coffee Lab](https://onyxcoffeelab.com), a championship-winning specialty roaster using the Loring S70 Peregrine roaster.

**Dataset Statistics:**
- **Profiles collected**: 49 (as of Oct 31, 2024, growing daily)
- **Temporal range**: Oct 30-31, 2024
- **Resolution**: 1-second intervals
- **Duration**: 9-15 minutes per profile
- **Roasting style**: High-charge, light-to-medium roasts
- **Unique products**: 36+ single origins and blends
- **Validation rate**: 100% (all profiles pass physics checks)

**Feature Coverage:**
| Feature | Coverage | Examples |
|---------|----------|----------|
| Origin | 100% | Ethiopia, Colombia, Kenya, Guatemala |
| Process | 100% | Washed, Natural, Honey, Anaerobic |
| Roast Level | 100% | Expressive Light, Medium, Dark |
| Variety | 95% | Heirloom, Caturra, Bourbon, Geisha |
| Altitude | 75% | 1200-2300 MASL |
| Flavor Notes | 100% | Berries, Chocolate, Floral, Citrus |

### **Data Collection Pipeline**
```bash
# Automatic batch tracking - won't duplicate profiles
python onyx_dataset_builder_v3_3_COMBINED.py

# Output: Date-stamped directory with new profiles
# onyx_dataset_2025_10_31/
#   ├── profiles/
#   │   ├── geometry_batch93253.json
#   │   └── monarch_batch93240.json
#   ├── complete_dataset.json
#   └── dataset_summary.csv
```

---

## 🏗️ Model Architecture

### **RoastFormer: Decoder-Only Transformer**

```
Input Features (17 dimensions)
    ├── Categorical (5)
    │   ├── Origin (embedding)
    │   ├── Process (embedding)
    │   ├── Roast Level (embedding)
    │   └── Variety (embedding)
    │
    ├── Continuous (4)
    │   ├── Target Finish Temp (normalized)
    │   ├── Altitude (normalized)
    │   ├── Bean Density Proxy (normalized)
    │   └── Caffeine Content (normalized)
    │
    └── Flavor (variable)
        └── Flavor Embeddings (averaged)

                    ↓
        
    Conditioning Module
    (Projects to d_model dimension)

                    ↓

    Transformer Decoder Layers
    ├── Multi-Head Self-Attention
    ├── Cross-Attention (with condition)
    ├── Feed-Forward Network
    └── Layer Normalization

                    ↓

    Output Projection
    (Temperature at next timestep)

                    ↓

    Auto-regressive Generation
    (426°F → 425°F → 424°F → ... → 395°F)
```

### **Model Configurations**

| Size | d_model | Heads | Layers | Parameters | Use Case |
|------|---------|-------|--------|------------|----------|
| Small | 128 | 4 | 4 | ~2M | Fast experiments |
| Medium | 256 | 8 | 6 | ~10M | Recommended |
| Large | 512 | 8 | 8 | ~40M | High quality |

### **Training Details**
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: AdamW (lr=1e-4)
- **Scheduler**: Cosine annealing
- **Batch size**: 16
- **Teacher forcing**: Yes (during training)
- **Max sequence length**: 1000 timesteps (~16 minutes)

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8+
- CUDA 11.0+ (optional, for GPU training)

### **Setup**

```bash
# Clone repository
git clone https://github.com/CKraiss18/roastformer.git
cd roastformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Requirements**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
selenium>=4.0.0
beautifulsoup4>=4.12.0
scikit-learn>=1.3.0
```

---

## 💻 Usage

### **1. Collect Dataset**

```bash
# Scrape Onyx Coffee Lab profiles (additive, won't duplicate)
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py

# Output: onyx_dataset_2024_MM_DD/
```

### **2. Explore Data**

```python
import pandas as pd

# Load dataset
df = pd.read_csv('onyx_dataset_2024_11_03/dataset_summary.csv')

# Check feature coverage
print(df[['origin', 'process', 'roast_level', 'flavor_notes_raw']].head())

# Analyze flavor distributions
flavors = df['flavor_notes_parsed'].str.split(', ').explode()
print(flavors.value_counts().head(10))
```

### **3. Train Model**

```python
from src.model import RoastFormer
from src.train import train_roastformer

# Load data
dataset = RoastProfileDataset('onyx_dataset_2024_11_03/')

# Initialize model
model = RoastFormer(
    conditioning_module=conditioning_module,
    d_model=256,
    nhead=8,
    num_layers=6
)

# Train
model = train_roastformer(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    device='cuda'
)
```

### **4. Generate Profiles**

```python
# Define conditions
conditions = {
    'origin': 'Ethiopia',
    'process': 'Washed',
    'variety': 'Heirloom',
    'roast_level': 'Light',
    'target_finish_temp': 395,
    'altitude': 2000,
    'flavors': ['berries', 'floral', 'citrus']  # ← Flavor guidance!
}

# Generate profile
generated_profile = model.generate(
    categorical_indices=encode_categorical(conditions),
    continuous_features=encode_continuous(conditions),
    flavor_notes=conditions['flavors'],
    start_temp=426.0,
    target_duration=600
)

# Visualize
plot_profile(generated_profile, title="Ethiopian Light Roast - Berries/Floral")
```

---

## 📁 Project Structure

```
roastformer/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Exclude large files
├── CLAUDE.md                           # Complete project instructions
├── TRAINING_PIPELINE_READY.md          # Infrastructure guide
│
├── src/                                # Source code (production-ready!)
│   ├── dataset/
│   │   ├── data_preparation.py         # Data loading & encoding
│   │   └── onyx_scraper.py             # Web scraper (v3.3)
│   ├── model/
│   │   └── roastformer.py              # Complete transformer architecture
│   ├── training/
│   │   ├── train.py                    # Training pipeline
│   │   └── evaluate.py                 # Evaluation pipeline
│   └── utils/
│       ├── validation.py               # Physics-based checks
│       ├── metrics.py                  # MAE, DTW, RoR metrics
│       └── visualization.py            # Profile plotting
│
├── notebooks/                          # Jupyter notebooks (coming soon)
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_results_visualization.ipynb
│
├── onyx_dataset_2025_10_30/           # Data (not in git)
│   ├── profiles/                       # 36 validated profiles
│   ├── complete_dataset.json
│   └── dataset_summary.csv
│
└── onyx_dataset_2025_10_31/           # Data (not in git)
    ├── profiles/                       # 13 validated profiles
    ├── complete_dataset.json
    └── dataset_summary.csv
```

---

## 📈 Results

### **Baseline Performance** (As of Nov 2024)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Temperature MAE | <5°F | TBD | 🔄 In progress |
| DTW Distance | <50 | TBD | 🔄 In progress |
| Monotonicity | 100% | TBD | 🔄 In progress |
| Bounded Rates | >95% | TBD | 🔄 In progress |
| Finish Temp Accuracy | >90% | TBD | 🔄 In progress |

### **Example Generations**

*Coming soon - profiles generated for Ethiopian Washed (light), Colombian Natural (medium), etc.*

### **Ablation Studies**

Planned experiments:
1. **Positional encoding variants** (Sinusoidal vs. Learned vs. RoPE)
2. **Flavor conditioning** (With vs. without flavor embeddings)
3. **Model size** (Small vs. Medium vs. Large)
4. **Conditioning features** (Phase 1 only vs. Phase 1+2 vs. Full)

---

## 🗺️ Roadmap

### **Phase 1: Baseline Implementation** ✅ **COMPLETE** (Oct 28 - Oct 31)
- [x] Dataset collection pipeline
- [x] Feature extraction (17 features)
- [x] Transformer architecture
- [x] Complete src/ infrastructure
- [x] Physics-based validation pipeline
- [x] Metrics & visualization modules
- [x] Training & evaluation pipelines
- [x] 49 validated profiles collected

### **Phase 2: Data Collection & Integration** 🔄 **IN PROGRESS** (Nov 1-8)
- [x] Continuous data collection (49 → target 80-100 profiles)
- [ ] Final data integration in training pipeline
- [ ] First baseline training run
- [ ] Initial validation results

### **Phase 3: Experiments & Optimization** (Nov 9-15)
- [ ] Baseline model training
- [ ] Flavor embedding experiments
- [ ] Positional encoding variants
- [ ] Model size ablation studies
- [ ] Hyperparameter tuning

### **Phase 4: Final Validation & Analysis** (Nov 16-22)
- [ ] Comprehensive evaluation
- [ ] Error analysis
- [ ] Comparison with real profiles
- [ ] Attention pattern visualization
- [ ] Success criteria validation

### **Phase 5: Capstone Completion** (Nov 23-30)
- [ ] Final report writing
- [ ] Presentation materials
- [ ] Model Card documentation
- [ ] Code cleanup & documentation
- [ ] Defense preparation

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{kraiss2024roastformer,
  title={RoastFormer: Transformer-Based Coffee Roast Profile Generation with Flavor Conditioning},
  author={Kraiss, Charlee},
  year={2024},
  school={[Vanderbilt University]},
  type={Master's Capstone Project}
}
```

---

## 🙏 Acknowledgments

- **Onyx Coffee Lab** - For publicly sharing roast profiles and inspiring this work
- **Capstone Advisor** - For guidance and support
- **PyTorch Team** - For the deep learning framework
- **Selenium & BeautifulSoup** - For web scraping capabilities

---

## 📞 Contact

**Charlee Kraiss**
- GitHub: [@CKraiss18](https://github.com/CKraiss18)
- Project: [RoastFormer](https://github.com/CKraiss18/roastformer)

---

## 🎓 About

This project is part of a Master's capstone project, Fall 2024.

**Project Goals:**
1. Demonstrate transformer application to time-series generation
2. Integrate domain knowledge (coffee science) into deep learning
3. Create practical tool for specialty coffee industry
4. Contribute novel approach to flavor-conditioned generation

---

## ⚡ Quick Start

```bash
# Clone & install
git clone https://github.com/CKraiss18/roastformer.git
cd roastformer
pip install -r requirements.txt

# Collect data
python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py

# Train model (coming soon)
python src/train.py --config configs/baseline.yaml

# Generate profile (coming soon)
python src/generate.py --origin Ethiopia --flavors "berries,floral"
```

---

## 📝 Development Log

### Recent Updates

**Oct 31, 2024 - Infrastructure Complete! 🎉**
- ✅ Built complete production-ready training infrastructure (Option B)
- ✅ Created src/ directory with proper organization
- ✅ Implemented physics-based validation (100% pass rate on 49 profiles)
- ✅ Built comprehensive metrics module (MAE, DTW, RoR, correlation)
- ✅ Created visualization tools (profile plots, training curves)
- ✅ Developed complete training pipeline with checkpointing
- ✅ Built evaluation pipeline with autoregressive generation
- ✅ Collected 49 validated profiles (36 from Oct 30 + 13 from Oct 31)
- ✅ Enhanced scraper (v3.3) with better error handling
- ✅ Pushed 6,144 lines of code to GitHub

**Oct 30, 2024 - Data Collection & Debugging**
- ✅ Fixed scraper issues (19 → 36 profiles)
- ✅ Increased chart load wait times
- ✅ Added non-coffee product filtering
- ✅ Validated all profiles against physics constraints

**Oct 28, 2024**
- ✅ Initial dataset collection
- ✅ Basic transformer architecture implemented
- ✅ Feature extraction pipeline complete

---

**Built with ☕ and 🤖**

*"Good coffee requires good data."*

---

*Last updated: October 31, 2024*
