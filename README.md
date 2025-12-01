# RoastFormer: Flavor-Conditioned Coffee Roast Profile Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Transformer-based generative model for coffee roast profiles, conditioned on bean characteristics and desired flavor outcomes**

**Fall 2025 | Generative AI Theory | Vanderbilt University**

---

## 📊 Quick Results

| Metric | Value | Significance |
|--------|-------|--------------|
| **Best Model RMSE** | 10.4°F | d=256 (6.4M params on 144 samples) |
| **Flavor Improvement** | +14% | Novel contribution validated |
| **Model Size** | 6,376,673 params | 51,843:1 param/sample ratio |
| **Training Dataset** | 144 profiles | Onyx Coffee Lab (2019 US Champions) |
| **PE Comparison** | Sinusoidal > RoPE | Validated on small data |

**Key Finding**: Normalization was critical (27x faster convergence). Largest model (d=256) performed best despite high parameter-to-sample ratio, disproving overfitting hypothesis with proper regularization.

---

## 📋 Table of Contents

1. [Problem Statement & Overview](#problem-statement--overview)
2. [Methodology](#methodology)
3. [Implementation & Demo](#implementation--demo)
4. [Assessment & Evaluation](#assessment--evaluation)
5. [Model & Data Cards](#model--data-cards)
6. [Critical Analysis](#critical-analysis)
7. [Documentation & Resource Links](#documentation--resource-links)

---

## 1. Problem Statement & Overview

### Coffee Journey: Seed to Cup

![Seed to Cup](seed-to-cup.jpeg)
*Roasting transforms green beans into aromatic coffee through controlled heating, developing 800+ flavor compounds. This critical 10-15 minute process determines final cup quality.*

---

### The Real-World Problem

**Coffee roasters spend 10-20 experimental roasts (~15 minutes each) per new coffee** to find an optimal profile—representing **2-3 hours and $200+ in wasted beans and labor** per coffee introduction.

Current methods rely on intuition, simple templates, and trial-and-error with no data-driven guidance for starting profiles conditioned on desired flavors.

---

### Why Transformers for Coffee Roasting?

Coffee roasting is a **sequential generation problem** where each temperature measurement is a **"token"** in a sequence. Like language models predicting the next word, RoastFormer predicts the next temperature given previous temperatures—but with physical constraints:

- **Token** = Temperature at time t (e.g., 426.2°F → 425.8°F → ...)
- **Sequence** = 400-1000 temperature tokens per profile
- **Context** = All previous temperatures inform next prediction
- **Conditioning** = Bean characteristics + desired flavors guide generation

**Key constraints**: Multi-modal features (categorical + continuous + multi-hot), physics laws (monotonicity, bounded heating rates), small data (144 samples), domain-specific evaluation beyond RMSE.

---

### Novel Contribution: Flavor-Conditioned Generation

**First transformer model to condition roast profile generation on desired flavor outcomes** (e.g., "berries", "chocolate", "floral"). Validated with **+14% performance improvement** over no-flavor baseline.

---

### Data Sourcing: Web Scraping Onyx Coffee Lab

![Onyx Scraping](onyx_data_scrape.png)
*Web scraping process: Collected profiles from Onyx Coffee Lab website over 8 scraping sessions (Oct 30 - Nov 10, 2025)*

![Profile Example](onyx_roast_profile.png)
*Example scraped profile showing temperature curve and rate-of-rise (RoR)*

Built custom scraper with **additive batch tracking** (no duplicates across runs):
- 8 scraping sessions → 159 files → **144 unique profiles** (123 train, 21 val)
- Source: [Onyx Coffee Lab](https://onyxcoffeelab.com) (2019 US Roaster Champions)
- Data: 1-second resolution temperature sequences + metadata (origin, process, variety, altitude, flavors)

---

## 2. Methodology

### Architecture Design

**Model**: Decoder-Only Transformer (Autoregressive Generation)

We chose a decoder-only architecture because roast profiles exhibit unidirectional causality—temperature at time t+1 depends on temperatures at t, t-1, and earlier, but not future time steps. This matches the causal structure of the physical roasting process.

**Best Configuration (d=256)**:
- 6 layers, 256 hidden dim, 8 heads
- 6.4M parameters total
- Sinusoidal positional encoding
- Dropout: 0.1, Weight decay: 0.01

---

### Normalization: The Critical Discovery

**Initial Failure**: All models predicted constant 16°F.

**Root Cause**: Networks output ~0-10 scale, we asked for 150-450°F temperatures → gradients exploded/vanished.

**Solution**: Normalize temperatures to [0,1] range → **27x faster convergence**. All models succeeded after normalization.

**Lesson**: Proper input/output scaling is essential for gradient flow, not optional optimization.

---

### Multi-Modal Conditioning Architecture

The model conditions on **17 features** across three modalities:

**Categorical (5)**: Origin (20 classes), Process (6), Variety (15), Roast Level (4), **Flavors (40 unique, multi-hot)**

**Continuous (4)**: Target finish temp, Altitude, Bean density, Caffeine content

The cross-attention mechanism allows the model to selectively attend to different conditioning features at each time step, learning which bean characteristics and flavor targets are relevant for predicting each temperature value.

---

### Positional Encoding: Empirical Comparison

| Method | Val RMSE | Notes |
|--------|----------|-------|
| **Sinusoidal** | **23.4°F** ✅ | Classic method (Vaswani 2017) |
| RoPE | 28.1°F | Rotary embeddings |
| Learned | 43.8°F | Overfits on 144 samples |

**Motivation**: Tested RoPE (Rotary Position Embeddings) based on my in-class paper presentation—curious whether rotating positions would help with time series prediction where temporal order is critical.

**Finding**: Simpler sinusoidal encodings outperformed RoPE on small data—deterministic patterns generalize better than learned or rotational methods in limited-data regimes.

---

### Flavor Conditioning: Validating the Novel Contribution

| Configuration | Val RMSE | Improvement |
|---------------|----------|-------------|
| Without flavors | 27.2°F | Baseline |
| **With flavors** | **23.4°F** | **+14% better** ✅ |

**What the Model Learns**: By conditioning on flavor notes like "berries", "chocolate", "floral", the model learns associations between temperature trajectories and sensory outcomes. For example:
- Berry flavors → certain development patterns (specific RoR curves)
- Chocolate notes → different temperature progressions
- Floral characteristics → distinct heating profiles

This validates that **task-relevant conditioning** improves generation beyond just bean metadata.

---

### Small-Data Strategies: Challenging the Overfitting Hypothesis

**Hypothesis**: "6.4M parameters will overfit on 123 samples."

| d_model | Params | Val RMSE | Params/Sample |
|---------|--------|----------|---------------|
| 32 | 203K | 43.8°F | 1,650:1 |
| 64 | 606K | 23.4°F | 4,925:1 |
| 128 | 2.0M | 16.5°F | 16,625:1 |
| **256** | **6.4M** | **10.4°F** ✅ | **51,843:1** |

**Result**: Largest model won! With normalization + regularization (dropout 0.1, weight decay 0.01, early stopping), capacity enables learning complex roast dynamics. Being experimentally wrong taught more than theory.

---

### Autoregressive Generation & Exposure Bias

**Challenge**: Model trained with teacher forcing (sees real temps) struggles generating independently (sees own predictions → errors compound).

**Evidence**: Training RMSE 10.4°F | Generation MAE 25.3°F (**2.4x degradation**)

**Physics Compliance Failures**:

What roast profiles **must follow**:
- ✅ **Monotonicity**: Temperature only increases after turning point (no cooling mid-roast)
- ✅ **Bounded heating rate**: 20-100°F/min (no scorching or baking)
- ✅ **Smooth transitions**: No sudden jumps (equipment limitation)

What our model **achieved**:
- ❌ Monotonicity: **0%** (profiles cool mid-roast—physically impossible)
- ⚠️ Bounded RoR: **28.8%** (heating rates too fast/slow)
- ✅ Smooth: **98.7%** (respected equipment constraints)

This is **exposure bias**—models not exposed to their own errors during training fail when generating autonomously.

---

### Domain-Specific Evaluation: Beyond Generic Metrics

**Generic metrics** (misleading): Training RMSE 10.4°F ✅, Generation MAE 25.3°F ⚠️

**Domain metrics** (revealing): Monotonicity 0% ❌, Bounded RoR 28.8% ⚠️, Smooth transitions 98.7% ✅

**Insight**: Standard metrics said "reasonable," physics metrics revealed "invalid profiles." Domain applications require domain-specific validation—understanding physical constraints is essential for proper evaluation.

---

## 3. Implementation & Demo

### Code Structure

The implementation consists of three main components:

**1. Data Preparation** (`src/dataset/preprocessed_data_loader.py`)
- Web scraping from Onyx Coffee Lab
- Feature extraction (categorical, continuous, flavors)
- Train/validation split (85%/15%)
- Data normalization and encoding

**2. Model Architecture** (`src/model/transformer_adapter.py`)
- Multi-modal conditioning module
- Decoder-only transformer blocks
- Cross-attention mechanism
- Autoregressive generation

**3. Training Pipeline** (`train_transformer.py`)
- AdamW optimizer with cosine annealing
- Early stopping with patience=20
- Gradient clipping and weight decay
- Checkpoint saving

### Usage Example

```python
from src.model.transformer_adapter import TransformerAdapter

# Load trained model
model = TransformerAdapter.from_pretrained('checkpoints/best_model_d256_epoch42.pt')

# Generate roast profile
profile = model.generate(
    origin='Ethiopia',
    process='Washed',
    roast_level='Expressive Light',
    flavors=['berries', 'floral', 'citrus'],
    target_finish_temp=395,
    altitude=2100,
    start_temp=426,
    target_duration=11*60  # 11 minutes
)

# Validate physics
from src.utils.validation import validate_physics
is_valid = validate_physics(profile)
```

### Interactive Notebooks

**Training Suite** (with Colab outputs): [`RoastFormer_Training_Suite_COMPREHENSIVE.ipynb`](RoastFormer_Training_Suite_COMPREHENSIVE.ipynb)
- Complete training experiments (7 ablations)
- Model size comparison (d=32, 64, 128, 256)
- Positional encoding ablation (sinusoidal, RoPE, learned)
- Flavor conditioning validation
- All cells executed with outputs visible

**Evaluation Demo** (with Colab outputs): [`RoastFormer_Evaluation_Demo_COMPLETE.ipynb`](RoastFormer_Evaluation_Demo_COMPLETE.ipynb)
- Generate profiles from validation set
- Compute evaluation metrics
- Visualize real vs generated comparisons
- Physics compliance analysis
- Interactive profile generation demo

Both notebooks include complete outputs from Google Colab training runs and can be viewed directly on GitHub.

---

## 4. Assessment & Evaluation

### Training Success: Comprehensive Ablation Studies

We conducted 7 systematic experiments to validate design choices and understand model behavior:

**Model Size Ablation**:

![Comprehensive Training Analysis](roastformer_COMPREHENSIVE_20251120_152131/comprehensive_analysis.png)
*Complete ablation study results showing model size comparison, positional encoding comparison, and flavor conditioning validation. The d=256 model achieved 10.4°F RMSE despite 51,843:1 parameter-to-sample ratio.*

**Key Results**:

| Experiment | Best Result | Finding |
|------------|-------------|---------|
| **Model Size** | d=256: 10.4°F | Larger model won (surprising!) |
| **Positional Encoding** | Sinusoidal: 23.4°F | Classic > modern on small data |
| **Flavor Conditioning** | +14% improvement | Novel contribution validated |

**Training Metrics** (d=256 best model):
- Final Validation RMSE: 10.4°F
- Training RMSE: 8.7°F  
- Convergence: 42 epochs (early stopping at 62)
- Training time: <20 minutes per experiment (GPU)

---

### Evaluation Results: Generation Quality & Challenges

**Real vs Generated Profiles**:

![Real vs Generated Comparison](roastformer_EVALUATION_20251120_170612/real_vs_generated_profiles.png)
*Comparison of 6 validation samples: real profiles (blue) vs generated profiles (orange). Generated profiles follow overall trajectory but exhibit physics violations (non-monotonic segments).*

**Quantitative Metrics** (10 validation samples):

| Metric | Value | Assessment |
|--------|-------|------------|
| **Temperature MAE** | 25.3°F | Reasonable accuracy |
| **RMSE** | 29.8°F | 2.9x worse than training |
| **Finish Temp MAE** | 13.95°F | Decent endpoint accuracy |
| **Finish Temp (±10°F)** | 50% | Half within tolerance |

**Physics Compliance** (reveals challenge):

| Constraint | Value | Status |
|------------|-------|--------|
| **Monotonicity** | 0.0% | ❌ All violate |
| **Bounded RoR** | 28.8% | ⚠️ Most out of bounds |
| **Smooth Transitions** | 98.7% | ✅ Good |
| **Overall Valid** | 0.0% | ❌ None pass |

**Analysis**: While temperature accuracy is reasonable (25°F MAE), generated profiles violate physical constraints. This identifies **autoregressive exposure bias** as the core challenge—model trained with teacher forcing struggles when generating independently.

---

### Detailed Profile Analysis

![Detailed Comparison](roastformer_EVALUATION_20251120_170612/detailed_comparison.png)
*Detailed view of a single profile showing temperature trajectory (top) and Rate of Rise (bottom). Generated profile (orange) follows general shape but lacks proper turning point physics.*

**Observations**:
- ✅ Overall trajectory shape captured
- ✅ Start and finish temperatures reasonable
- ❌ Turning point dynamics incorrect (should dip, then recover)
- ❌ RoR pattern unrealistic (should show characteristic phases)
- ❌ Non-monotonic segments mid-roast (physically impossible)

---

### Example Use Cases: Diverse Coffee Profiles

![Example Use Cases](roastformer_EVALUATION_20251120_170612/example_use_cases.png)
*Four diverse generation examples: Ethiopian light roast (berry/floral), Colombian medium (chocolate/nutty), Kenyan bright (citrus/winey), and Guatemala balanced. Model generates distinct profiles for different bean characteristics and flavor targets.*

**Diversity Analysis**:
- Different origins → different temperature progressions
- Flavor targets influence RoR patterns  
- Roast levels affect finish temperature and development time
- Model learned meaningful bean characteristic associations

---

### Interactive Demo Results

![Demo Profile](roastformer_EVALUATION_20251120_170612/demo_profile.png)
*Custom profile generated for Ethiopian washed coffee targeting "berries, floral, citrus" at 395°F finish. Shows temperature curve, RoR, and key roast phases. Despite physics violations, demonstrates controllable generation from user specifications.*

**Demo Features**:
- User specifies all conditioning features
- Model generates complete profile in <1 second
- Visualization shows temp + RoR curves
- Metrics computed automatically
- Physics validation provides feedback

---

## 5. Model & Data Cards

*Rubric Requirements: • Model version/architecture is shown • Intended uses and licenses is outlined • Ethical/bias considerations are addressed*

### Model Card Summary

**Full Details**: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)

**Model Version/Architecture** *(Rubric Required)*:

| Attribute | Value |
|-----------|-------|
| **Model Name** | RoastFormer v1.0 |
| **Architecture** | Decoder-only Transformer |
| **Parameters** | 6,376,673 (d=256, 6 layers, 8 heads) |
| **Training Data** | 144 Onyx profiles (123 train, 21 val) |
| **Best RMSE** | 10.4°F (validation) |
| **Novel Contribution** | Flavor-conditioned generation (+14% improvement) |
| **Positional Encoding** | Sinusoidal (Vaswani et al. 2017) |

**Intended Uses and Licenses** *(Rubric Required)*:

**Intended Use**:
- ✅ Generate starting roast profiles for new coffees
- ✅ Explore "what-if" scenarios (different origins, processes, flavors)
- ✅ Reduce experimentation time from 10-20 roasts to 2-3 refinements
- ✅ Research and education in coffee science and ML applications

**License**:
- **Code**: MIT License (free to use, modify, distribute)
- **Documentation**: CC BY-NC 4.0 (attribution required, non-commercial)
- **Model Weights**: Available for research/education use

**Out-of-Scope Uses**:
- ❌ Production roasting without human validation (0% physics compliance)
- ❌ Commodity coffee (trained on specialty-grade only)
- ❌ Equipment outside 10-50 lb batch range (Loring S70 specific)

**Ethical/Bias Considerations** *(Rubric Required)*:

**Ethical Considerations**:
- ✅ Data sourced from public profiles (Onyx Coffee Lab) with full attribution
- ⚠️ Model learns "Onyx's championship style" not general roasting practices
- ⚠️ Requires expert validation before use (physics violations present)
- ✅ Open source (MIT license) promotes transparency and reproducibility

**Bias Considerations**:
- **Single-roaster bias**: Model trained exclusively on Onyx data → may not generalize to other roasters' styles or equipment
- **Geographic bias**: 48% African/Central American origins → underrepresents Asian coffees
- **Roast level bias**: 72% light roasts → may generate poor dark roast profiles
- **Equipment bias**: Loring S70 convection roaster only → patterns may not transfer to drum roasters

---

### Data Card Summary

**Full Details**: [`docs/DATA_CARD.md`](docs/DATA_CARD.md)

| Attribute | Value |
|-----------|-------|
| **Dataset** | Onyx Coffee Lab Roast Profiles |
| **Size** | 144 profiles (123 train, 21 val) |
| **Temporal Coverage** | October-November 2025 |
| **Resolution** | 1-second intervals (400-1000 time steps) |
| **Geographic Coverage** | 20+ origins (Ethiopia 29%, Colombia 19%) |
| **Equipment** | Loring S70 Peregrine (convection roaster) |

**Features Extracted**:
- 5 categorical: origin, process, variety, roast level, flavors
- 4 continuous: target temp, altitude, density, caffeine
- 1 time-series: temperature sequence (1-second resolution)

**Known Biases**:
- **Single-roaster bias** (critical): All from Onyx → learns their "house style"
- **Light roast bias**: 72% light, 23% medium, 5% dark
- **Geographic bias**: African/Central American heavy (48%)
- **Modern equipment**: Loring S70 only (no drum roasters)

**Critical Limitation**: Even 500+ Onyx profiles wouldn't fix single-roaster bias. Need 10+ diverse roasters (different equipment, styles, regions) for true generalization.

**Ethical Data Collection**:
- ✅ Public data only (no login required)
- ✅ Rate-limited scraping (respectful)
- ✅ Full attribution to Onyx Coffee Lab
- ✅ Research/education use (non-commercial)

---

## 6. Critical Analysis

*Rubric Requirements: Answered one or more of the following questions: **What is the impact of this project?** | **What does it reveal or suggest?** | **What is the next step?***

---

### What does this project reveal or suggest? *(Rubric Question)*

**Key Insight 1: The Normalization Discovery**

Initial complete failure (all models predicting constant 16°F) led to systematic debugging that revealed a fundamental principle: networks need proper input/output scaling for gradient flow. The 27x convergence speedup after normalization wasn't just an optimization—it was the difference between complete failure and success.

**What this reveals**: Understanding why something fails teaches more than knowing it works. This debugging process demonstrated the importance of analyzing training dynamics, not just trying different hyperparameters.

**Key Insight 2: The d=256 Surprise**

We predicted the 6.4M parameter model would overfit on 123 samples. It achieved the best results.

**What this reveals**: Being experimentally wrong revealed that modern regularization techniques (dropout, weight decay, early stopping) combined with proper normalization enable large models to work in small-data regimes. Theoretical assumptions about overfitting were overturned by empirical evidence.

**Lesson**: Run the experiment even when you "know" it won't work. Empirical validation beats assumptions.

**Key Insight 3: The Limits of Post-Processing**

**Attempted Solution: Physics-Constrained Generation**

To address 0% physics compliance, we implemented physics constraints during generation:

![Constrained vs Unconstrained](roastformer_EVALUATION_20251120_170612/constrained_vs_unconstrained_comparison.png)
*Comparison of unconstrained generation (left) vs physics-constrained generation (right). Constrained approach enforced monotonicity and bounded RoR but resulted in unrealistic linear ramps and 4.5x worse accuracy.*

**Results**: FAILED

| Metric | Unconstrained | Constrained | Change |
|--------|---------------|-------------|--------|
| **MAE** | 25.3°F | 113.6°F | **+88.3°F (4.5x worse)** ❌ |
| **Finish Temp MAE** | 13.95°F | 86.67°F | **+72.7°F worse** ❌ |
| **Monotonicity** | 0.0% | 100.0% | +100% ✅ |
| **Bounded RoR** | 28.8% | 0.0% | **-28.8% (worse!)** ❌ |

**Why It Failed**:

The constraints fought against the model's learned behavior. During training with teacher forcing, the model learned temperature patterns that occasionally include non-monotonic segments, unbounded heating rates, and complex dynamics. Post-generation constraints tried to force physical behavior the model never learned, resulting in unnatural linear ramps instead of realistic curves.

**Root Cause**: Post-processing cannot fix training issues. The model was trained to mimic training sequences (with teacher forcing), not to generate physically valid sequences independently.

**What this reveals**: Solutions must address the root cause—the training process—not the symptoms. Attempting to "fix" generation output reveals fundamental misunderstanding of where the problem originates.

---

### What is the next step? *(Rubric Question)*

Based on this analysis and literature review, proper solutions require training-time fixes:

**1. Scheduled Sampling** (Bengio et al., 2015)
- Gradually transition from teacher forcing to model predictions during training
- Model learns to handle its own prediction errors
- Addresses exposure bias at the source

**2. Physics-Informed Loss Functions**
- Add penalty terms for physics violations to training loss
- Model learns constraints, not just patterns
- Example: `loss = mse_loss + λ₁*monotonicity_penalty + λ₂*ror_penalty`

**3. Multi-Roaster Dataset** (Most Critical!)
- **Not just more Onyx data** - need 10+ diverse roasters
- Equipment diversity: Loring, Probat, Diedrich, Giesen (drum), Sivetz (fluid bed)
- Style diversity: Nordic light, traditional medium, French dark, espresso
- Geographic diversity: US, Europe, Asia, Africa roasting cultures
- **Key insight**: Diversity > scale. 200 profiles from 10 roasters > 500 from one roaster

**4. Duration Prediction Module**
- Current: User specifies duration (design choice)
- Future: Model predicts optimal duration
- "This dense Ethiopian at 2100m needs 11.5 min for light roast"

**5. Non-Autoregressive Architectures**
- Diffusion models for profile generation
- Generate entire sequence at once (no error accumulation)
- Eliminates exposure bias entirely

---

### What is the impact of this project? *(Rubric Question)*

**For Specialty Coffee**:
- Demonstrates feasibility of data-driven profile generation
- Validates flavor conditioning as meaningful feature (14% improvement)
- Identifies clear path forward with literature-backed solutions

**For ML Research**:
- Validates transformers for domain-specific physical processes
- Demonstrates small-data success (51,843:1 ratio) with proper techniques
- Shows importance of domain-specific evaluation (physics vs generic metrics)
- Provides instructive example of exposure bias in real application

**For AI Education**:
- Complete documentation of debugging process (normalization discovery)
- Honest reporting of failures (constrained generation)
- Systematic ablation studies (7 experiments)
- Clear connection between theory and practice

---

## 7. Documentation & Resource Links

### Repository Structure

```
roastformer/
├── README.md                                    # This file
├── docs/
│   ├── MODEL_CARD.md                           # Complete model documentation
│   ├── DATA_CARD.md                            # Dataset documentation
│   ├── EVALUATION_FINDINGS.md                  # Detailed evaluation analysis
│   ├── COMPREHENSIVE_RESULTS.md                # All training experiments
│   ├── METHODOLOGY_COURSE_CONNECTIONS.md       # Course concept mapping
│   ├── RUBRIC_COURSE_MAPPING.md               # Rubric alignment
│   └── FINAL_README_PRESENTATION_PLAN.md      # Presentation guide
├── src/
│   ├── dataset/
│   │   └── preprocessed_data_loader.py         # Data loading & encoding
│   ├── model/
│   │   └── transformer_adapter.py              # Model architecture
│   └── training/
├── train_transformer.py                        # Training pipeline
├── evaluate_transformer.py                     # Evaluation suite
├── generate_profiles.py                        # Profile generation
├── RoastFormer_Training_Suite_COMPREHENSIVE.ipynb      # Training experiments (with outputs)
├── RoastFormer_Evaluation_Demo_COMPLETE.ipynb          # Evaluation demo (with outputs)
├── roastformer_EVALUATION_20251120_170612/    # Evaluation results & images
└── roastformer_COMPREHENSIVE_20251120_152131/  # Training results & images
```

---

### Setup Instructions

**Requirements**:
```bash
Python >= 3.8
PyTorch >= 2.0.0
numpy >= 1.23.0
pandas >= 1.5.0
```

**Installation**:
```bash
# Clone repository
git clone https://github.com/CKraiss18/roastformer.git
cd roastformer

# Install dependencies
pip install -r requirements.txt
```

**Quick Start**:
```bash
# Generate a profile
python generate_profiles.py \
  --origin "Ethiopia" \
  --process "Washed" \
  --roast_level "Light" \
  --flavors "berries,floral,citrus" \
  --target_temp 395 \
  --altitude 2100

# Train model (requires preprocessed data)
python train_transformer.py --d_model 256 --num_layers 6

# Evaluate model
python evaluate_transformer.py --checkpoint checkpoints/best_model.pt
```

---

### Citations & References

**Key Literature**:

1. **Vaswani et al. (2017)** - "Attention is All You Need"
   - Transformer architecture foundation
   - Sinusoidal positional encodings (best on our small data)

2. **Bengio et al. (2015)** - "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
   - Proper solution to exposure bias
   - Gradually transition from teacher forcing to model predictions

3. **Su et al. (2021)** - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - RoPE positional encodings
   - Compared empirically (sinusoidal won on small data)

**Code & Resources**:
- Repository: https://github.com/CKraiss18/roastformer
- Onyx Coffee Lab: https://onyxcoffeelab.com (data source)
- PyTorch Transformers: https://pytorch.org/docs/stable/nn.html#transformer

---

### Citation

If you use RoastFormer in your work, please cite:

```bibtex
@software{kraiss2025roastformer,
  author = {Kraiss, Charlee},
  title = {RoastFormer: Flavor-Conditioned Coffee Roast Profile Generation with Transformers},
  year = {2025},
  institution = {Vanderbilt University},
  course = {Generative AI Theory (Fall 2025)},
  url = {https://github.com/CKraiss18/roastformer}
}
```

---

## Acknowledgments

**Data Source**: Onyx Coffee Lab (https://onyxcoffeelab.com)
- 2019 US Roaster Champions
- Thank you for publicly sharing roast profiles and advancing specialty coffee transparency

**Course Support**: Vanderbilt University Generative AI Theory (Fall 2025)
- Instructor and TAs for guidance on transformer implementation and evaluation methodology
- Course concepts applied: neural network fundamentals, transformers, conditional generation, small-data strategies, evaluation methodology

**Coffee Origins**: Gratitude to coffee farmers in Ethiopia, Colombia, Guatemala, Kenya, and beyond for producing exceptional coffees that make this research possible.

---

**Last Updated**: December 1, 2025
**Status**: Research Prototype (NOT production-ready - requires physics validation)
**License**: MIT (code), CC BY-NC 4.0 (documentation)
**Contact**: charlee.kraiss@vanderbilt.edu

