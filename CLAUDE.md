# RoastFormer Development Guide

**Project**: Transformer-Based Coffee Roast Profile Generation  
**Developer**: Charlee Kraiss  
**Timeline**: Oct 2024 - Nov 2024 (Capstone Project)  
**Repository**: https://github.com/CKraiss18/roastformer

---

## 🎯 Project Mission

Generate physically plausible coffee roast profiles using transformer architecture, conditioned on:
- Bean characteristics (origin, process, variety, altitude, density)
- Target roast level (light, medium, dark)
- **Desired flavor profile** (berries, chocolate, floral, etc.) ← Novel contribution

**Problem**: Roasters spend 10-20 experimental roasts (~15 min each) per new coffee, working from zero.  
**Solution**: RoastFormer provides data-driven starting profiles, validated against real specialty coffee data from Onyx Coffee Lab.

---

## 📊 Current Project Status (Nov 3, 2024)

### ✅ **Completed**
- [x] Dataset collection pipeline (Onyx scraper v3.3)
- [x] Feature extraction (17 features: categorical + continuous + flavors)
- [x] Transformer architecture design (decoder-only, ~400 lines)
- [x] Data preparation script (01_data_preparation.py)
- [x] Batch tracking system (additive, no duplicates)
- [x] Validation dataset: **28-36 real roast profiles** from Onyx Coffee Lab

### 🔄 **In Progress**
- [ ] Data validation & quality checks
- [ ] Integrate scraper output with transformer architecture
- [ ] Training pipeline implementation
- [ ] Baseline model training

### 📅 **Next Priority** (Nov 3-8)
1. **Data validation pipeline** - Ensure scraped data quality
2. **Feature encoding integration** - Connect data_prep to architecture
3. **Training loop** - Implement full train/val cycle
4. **Baseline experiments** - First model training run

---

## 🗂️ Repository Structure

```
roastformer/
├── CLAUDE.md                           # This file - your primary reference
├── README.md                           # Project overview, installation, usage
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Exclude datasets, checkpoints
│
├── docs/
│   ├── ARCHITECTURE_QUICK_REFERENCE.md # Transformer design summary
│   ├── FEATURE_EXTRACTION_GUIDE.md     # Feature engineering guide
│   └── Kraiss_Charlee_RoastFormer.pdf  # Capstone proposal
│
├── src/
│   ├── dataset/
│   │   ├── onyx_dataset_builder_v3_3_COMBINED.py  # Web scraper (LATEST)
│   │   └── 01_data_preparation.py                  # Data loading & encoding
│   │
│   ├── model/
│   │   └── ROASTFORMER_ARCHITECTURE_REFERENCE.py   # Complete model code
│   │
│   ├── training/
│   │   ├── train.py                    # Training loop (TO CREATE)
│   │   ├── evaluate.py                 # Validation metrics (TO CREATE)
│   │   └── callbacks.py                # Logging, checkpointing (TO CREATE)
│   │
│   └── utils/
│       ├── validation.py               # Physics-based checks (TO CREATE)
│       ├── visualization.py            # Plot profiles (TO CREATE)
│       └── metrics.py                  # MAE, DTW, etc. (TO CREATE)
│
├── tests/
│   ├── test_data_preparation.py        # Data loader tests
│   ├── test_model_architecture.py      # Model component tests
│   ├── test_training_pipeline.py       # Training tests
│   └── test_validation.py              # Physics constraint tests
│
├── scripts/
│   ├── run_scraper.sh                  # Collect new data
│   ├── validate_data.sh                # Check data quality
│   ├── train_baseline.sh               # Train model
│   └── evaluate_model.sh               # Run evaluation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Dataset analysis
│   ├── 02_feature_engineering.ipynb    # Feature design
│   ├── 03_baseline_training.ipynb      # Training experiments
│   └── 04_results_analysis.ipynb       # Results & visualizations
│
├── onyx_dataset_YYYY_MM_DD/           # Scraped data (NOT in git)
│   ├── profiles/*.json                 # Individual roast profiles
│   ├── complete_dataset.json           # Full dataset
│   └── dataset_summary.csv             # Feature matrix
│
├── preprocessed_data/                  # Processed data (NOT in git)
│   ├── training_data.pt                # Train/val split
│   └── dataset_stats.json              # Statistics
│
├── checkpoints/                        # Model weights (NOT in git)
│   └── baseline_*.pt
│
└── results/                            # Outputs (NOT in git)
    ├── generated_profiles/
    ├── visualizations/
    └── metrics/
```

---

## 🏗️ Technical Architecture

### **Model: RoastFormer (Decoder-Only Transformer)**

**Input**: Conditioning features → **Output**: Temperature sequence (autoregressive)

```
┌─────────────────────────────────────────┐
│  CONDITIONING INPUT (17 features)      │
├─────────────────────────────────────────┤
│ Categorical (5):                        │
│   • Origin (Ethiopia, Colombia, etc.)   │
│   • Process (Washed, Natural, Honey)    │
│   • Roast Level (Light, Medium, Dark)   │
│   • Variety (Heirloom, Caturra, etc.)   │
│   • Flavors (multi-hot encoding)        │
│                                          │
│ Continuous (4):                          │
│   • Target Finish Temp (normalized)     │
│   • Altitude (normalized)                │
│   • Bean Density Proxy (normalized)     │
│   • Caffeine Content (normalized)       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  CONDITIONING MODULE                    │
│  • Categorical → Embeddings (32-dim)    │
│  • Continuous → Linear projection       │
│  • Concatenate → Unified vector         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  TRANSFORMER DECODER (6 layers)         │
│  • Multi-head self-attention (8 heads)  │
│  • Cross-attention with conditioning    │
│  • Feed-forward networks                │
│  • Layer normalization                  │
│  • Positional encoding (3 variants)     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  OUTPUT PROJECTION                      │
│  • Linear: d_model → 1                  │
│  • Predicts temperature at next step    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  AUTOREGRESSIVE GENERATION              │
│  426°F → 425°F → 424°F → ... → 395°F   │
└─────────────────────────────────────────┘
```

### **Model Configurations**

| Size | d_model | Heads | Layers | Params | Use Case |
|------|---------|-------|--------|--------|----------|
| Small | 128 | 4 | 4 | ~2M | Fast experiments, debugging |
| Medium | 256 | 8 | 6 | ~10M | **Recommended baseline** |
| Large | 512 | 8 | 8 | ~40M | High quality (if data sufficient) |

### **Training Configuration**

```python
# Recommended baseline
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"
LOSS = "MSE"
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
```

---

## 🔬 Domain Knowledge: Coffee Roasting Physics

### **Valid Temperature Ranges**
- **Charge temp**: 400-450°F (initial roaster temperature)
- **Turning point**: 250-320°F (minimum temp after bean charge)
- **First crack**: 380-400°F (beans audibly pop)
- **Drop temp**: 
  - Light roast: 390-400°F (Agtron #125-145)
  - Medium roast: 405-415°F (Agtron #85-105)
  - Dark roast: 420-430°F (Agtron #55-75)

### **Valid Heating Rates**
- **Rate of Rise (RoR)**: 20-100°F/min
- **Too fast**: >100°F/min → scorching, uneven development
- **Too slow**: <20°F/min → baking, flat flavors

### **Profile Phases**
1. **Drying Phase** (0-4 min): 400°F → 320°F → 350°F
   - Beans lose moisture
   - Temperature drops then recovers
   
2. **Maillard Phase** (4-8 min): 350°F → 380°F
   - Browning reactions
   - Flavor compound development
   
3. **Development Phase** (8-12 min): 380°F → 395°F (drop)
   - Post-first-crack
   - Final flavor adjustments
   - Critical for roast level

### **Physics Constraints (For Validation)**
```python
# Monotonicity check (post-turning-point)
assert all(temps[i+1] >= temps[i] for i in range(turning_point_idx, len(temps)))

# Bounded heating rates
ror = np.diff(temps)
assert (ror >= 20/60).sum() / len(ror) > 0.95  # >95% within bounds
assert (ror <= 100/60).sum() / len(ror) > 0.95

# Finish temp accuracy
assert abs(temps[-1] - target_finish_temp) < 10  # Within 10°F

# No sudden jumps
assert (np.abs(np.diff(temps)) < 10/60).all()  # <10°F per second
```

### **Onyx Coffee Lab Specifics**
- **Roaster**: Loring S70 Peregrine (highly efficient, clean heat)
- **Batch size**: ~10-50 lbs
- **Typical duration**: 9-12 minutes
- **Charge temp**: 420-430°F
- **Style**: High-charge, fast development (modern light roasting)
- **Data resolution**: 1-second intervals

---

## 📦 Dataset Details

### **Onyx Validation Set**
- **Source**: https://onyxcoffeelab.com (championship-winning specialty roaster)
- **Profiles collected**: 28-36 (as of Nov 3, 2024)
- **Temporal range**: Oct-Nov 2024
- **Format**: JSON (individual profiles) + CSV (summary)
- **Collection method**: Automated web scraping with batch tracking

### **Feature Coverage**

| Feature | Coverage | Unique Values | Example |
|---------|----------|---------------|---------|
| Origin | 100% | 15-20 | "Colombia, Ethiopia" |
| Process | 100% | 4-6 | "Washed", "Natural", "Honey" |
| Roast Level | 100% | 3-4 | "Expressive Light", "Medium" |
| Variety | 95% | 10-15 | "Heirloom", "Mixed", "Caturra" |
| Altitude | 75% | continuous | 1200-2300 MASL |
| Flavors | 100% | 30-40 | "berries", "chocolate", "floral" |
| Bean Temp | 100% | 400-600 pts | 425.4°F → 404.5°F |
| RoR | 100% | 400-600 pts | Computed from bean_temp |

### **Data Quality Checks**

```python
# Temperature sequence validation
assert 350 < temps[0] < 450  # Valid charge temp
assert 380 < temps[-1] < 430  # Valid drop temp
assert 400 < len(temps) < 1000  # Reasonable duration (7-16 min)

# Metadata validation
assert pd.notna(row['origin'])
assert row['process'] in ['Washed', 'Natural', 'Honey', 'Anaerobic', 'Experimental']
assert 70 < row['roast_level_agtron'] < 150
assert 390 < row['target_finish_temp'] < 430

# Flavor validation
assert len(row['flavor_notes_parsed']) > 0
assert all(isinstance(f, str) for f in row['flavor_notes_parsed'])
```

---

## 🔧 Development Workflow (Git Flow)

### **Branch Strategy**

```
main
  ↓
develop (integration branch)
  ├── feature/data-validation
  ├── feature/training-pipeline
  ├── feature/baseline-model
  ├── feature/flavor-conditioning
  ├── experiment/positional-encodings
  ├── experiment/model-sizes
  └── bugfix/scraper-fixes
```

### **Branch Naming**
- `feature/` - New functionality (data prep, training loop, etc.)
- `experiment/` - ML experiments (ablations, hyperparameter tuning)
- `bugfix/` - Bug fixes
- `hotfix/` - Urgent fixes to main

### **Commit Message Format**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `experiment`

**Examples**:
```bash
feat(data): Add physics-based validation checks
test(model): Add unit tests for conditioning module
experiment(pos_enc): Compare sinusoidal vs RoPE
fix(scraper): Handle missing altitude values
```

### **Pull Request Process**
1. Create feature branch from `develop`
2. Make changes, add tests
3. Run all tests: `pytest tests/`
4. Push and create PR to `develop`
5. Review → Merge → Delete branch
6. Periodic merges: `develop` → `main` (production-ready code)

---

## 🧪 Testing Strategy (pytest)

### **Test Structure**

```
tests/
├── conftest.py                    # Shared fixtures
├── test_data_preparation.py       # Data loader tests
├── test_model_components.py       # Unit tests for modules
├── test_training_integration.py   # End-to-end training
└── test_physics_validation.py     # Domain constraint tests
```

### **Required Test Coverage**

**1. Data Preparation Tests** (`test_data_preparation.py`)
```python
def test_load_dataset():
    """Test dataset loading from Onyx directories"""
    assert len(profiles) > 0
    assert all('metadata' in p for p in profiles)

def test_feature_encoding():
    """Test categorical and continuous encoding"""
    assert categorical_indices['origin'] in range(num_origins)
    assert 0 <= continuous_features[0] <= 1  # normalized

def test_train_val_split():
    """Test data splitting preserves distribution"""
    assert len(train) + len(val) == len(dataset)
    assert val_ratio - 0.05 < len(val)/len(dataset) < val_ratio + 0.05
```

**2. Model Architecture Tests** (`test_model_components.py`)
```python
def test_conditioning_module():
    """Test feature encoding and projection"""
    condition = conditioning_module(categorical, continuous, flavors)
    assert condition.shape == (batch_size, embed_dim * 5)

def test_transformer_forward():
    """Test forward pass shape and values"""
    output = model(input_seq, condition)
    assert output.shape == (batch_size, seq_len, 1)
    assert torch.isfinite(output).all()

def test_autoregressive_generation():
    """Test generation produces valid profiles"""
    profile = model.generate(condition, start_temp=426, steps=600)
    assert 350 < profile[0] < 450  # Valid charge
    assert 380 < profile[-1] < 430  # Valid drop
```

**3. Physics Validation Tests** (`test_physics_validation.py`)
```python
def test_monotonicity():
    """Test post-turning-point monotonic increase"""
    turning_idx = np.argmin(temps)
    assert (np.diff(temps[turning_idx:]) >= 0).all()

def test_bounded_heating_rates():
    """Test heating rates within physical limits"""
    ror = np.diff(temps) * 60  # °F/min
    assert (ror >= 20).sum() / len(ror) > 0.95
    assert (ror <= 100).sum() / len(ror) > 0.95

def test_finish_temp_accuracy():
    """Test profiles reach target finish temp"""
    assert abs(temps[-1] - target_finish) < 10
```

**4. Training Integration Tests** (`test_training_integration.py`)
```python
def test_training_loop_runs():
    """Test training completes without errors"""
    model = train_roastformer(model, train_loader, val_loader, num_epochs=2)
    assert model is not None

def test_loss_decreases():
    """Test loss decreases over epochs"""
    losses = train_roastformer(..., return_losses=True)
    assert losses[-1] < losses[0]

def test_checkpoint_saving():
    """Test model checkpoints save correctly"""
    assert Path('checkpoints/model_epoch_1.pt').exists()
```

### **Running Tests**

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_preparation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run in parallel
pytest tests/ -n auto
```

### **Test Fixtures** (`conftest.py`)

```python
import pytest
import torch
from pathlib import Path

@pytest.fixture
def sample_profile():
    """Load a sample Onyx profile for testing"""
    with open('onyx_dataset_2024_11_03/profiles/geometry_batch93187.json') as f:
        return json.load(f)

@pytest.fixture
def sample_dataset():
    """Create small dataset for testing"""
    loader = RoastProfileDataLoader(auto_discover=True)
    profiles, metadata = loader.load_dataset()
    return profiles[:5], metadata.head(5)  # Just 5 samples

@pytest.fixture
def small_model():
    """Create small model for fast testing"""
    return RoastFormer(
        conditioning_module=conditioning_module,
        d_model=64,
        nhead=4,
        num_layers=2
    )

@pytest.fixture
def device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 📈 Success Metrics & Validation

### **Quantitative Metrics**

| Metric | Target | Priority | How to Check |
|--------|--------|----------|--------------|
| **Temperature MAE** | <5°F | Critical | `np.mean(np.abs(real - generated))` |
| **DTW Distance** | <50 | High | `dtw(real, generated)` |
| **Finish Temp Accuracy** | >90% | Critical | `abs(final - target) < 10` |
| **Monotonicity** | 100% | Critical | `all(diff >= 0) after turning point` |
| **Bounded RoR** | >95% | High | `(20 < ror < 100).mean()` |
| **Smooth Transitions** | 100% | High | `all(abs(diff) < 10/60)` |

### **Qualitative Assessments**

1. **Visual Inspection**: Plot real vs generated profiles
2. **Phase Recognition**: Does attention learn drying/maillard/development?
3. **Flavor Conditioning**: Do flavor-guided generations differ meaningfully?
4. **Generalization**: Test on unseen origins/processes

### **Validation Checklist**

```python
def validate_profile(real_profile, generated_profile, target_finish_temp):
    """Complete validation suite"""
    
    checks = {}
    
    # 1. Temperature accuracy
    mae = np.mean(np.abs(real_profile - generated_profile))
    checks['mae_ok'] = mae < 5.0
    
    # 2. DTW similarity
    dtw_dist = dtw_distance(real_profile, generated_profile)
    checks['dtw_ok'] = dtw_dist < 50
    
    # 3. Physics constraints
    turning_idx = np.argmin(generated_profile)
    checks['monotonic'] = (np.diff(generated_profile[turning_idx:]) >= 0).all()
    
    ror = np.diff(generated_profile) * 60
    checks['bounded_ror'] = ((ror >= 20) & (ror <= 100)).mean() > 0.95
    
    # 4. Target achievement
    checks['finish_temp'] = abs(generated_profile[-1] - target_finish_temp) < 10
    
    # 5. Smoothness
    checks['smooth'] = (np.abs(np.diff(generated_profile)) < 10/60).all()
    
    return checks, all(checks.values())
```

---

## 🎓 Capstone Timeline (Nov 2024)

### **Week 1: Nov 3-8** (Baseline Implementation)
- **Mon-Tue**: Data validation pipeline, integrate with architecture
- **Wed-Thu**: Training loop implementation, first baseline training
- **Fri**: Debug issues, validate initial results

**Deliverables**: 
- Working training pipeline
- Baseline model trained on Onyx data
- Initial validation metrics

### **Week 2: Nov 10-15** (Ablation Studies)
- **Mon-Tue**: Positional encoding experiments (sinusoidal, learned, RoPE)
- **Wed-Thu**: Model size experiments (small, medium, large)
- **Fri**: Conditioning feature ablations

**Deliverables**:
- Ablation study results
- Best configuration identified
- Attention pattern analysis

### **Week 3: Nov 17-22** (Final Validation & Presentation)
- **Mon-Tue**: Final model training, comprehensive evaluation
- **Wed**: Generate example profiles, create visualizations
- **Thu**: Finalize presentation materials, Model Card
- **Fri**: Practice presentation, final report

**Deliverables**:
- Final trained model
- Presentation materials
- Model Card
- Pseudocode documentation

---

## 🚨 Common Issues & Solutions

### **Issue: Small Dataset Overfitting**

**Problem**: Only 28-36 profiles → model memorizes training set

**Solutions**:
1. Use small model (d_model=128, 4 layers)
2. Heavy regularization (dropout=0.2, weight_decay=0.05)
3. Early stopping (patience=10)
4. Data augmentation (temperature jitter, time warping)
5. Physics-based constraints during generation

### **Issue: Profile Generation Diverges**

**Problem**: Generated temperatures go outside valid range

**Solutions**:
1. Clip outputs: `torch.clamp(output, min=250, max=450)`
2. Add physics loss term: `loss_physics = penalty_for_invalid_ror(output)`
3. Use constrained generation: reject invalid samples
4. Lower learning rate

### **Issue: Missing Features in Data**

**Problem**: Some profiles lack altitude, variety, etc.

**Solutions**:
1. Use defaults: `altitude = 1500` if missing
2. Origin-based imputation: `altitude['Ethiopia'] = 2000`
3. Skip profiles with too many missing features
4. Train separate model variants (with/without Phase 2 features)

### **Issue: Slow Training**

**Problem**: Training takes too long on CPU

**Solutions**:
1. Use smaller model for experiments
2. Reduce max_sequence_length: `800` instead of `1000`
3. Smaller batch size: `8` instead of `16`
4. Use GPU if available
5. Profile code to find bottlenecks

---

## 💻 Quick Reference Commands

### **Data Collection**
```bash
# Scrape new Onyx profiles
python src/dataset/onyx_dataset_builder_v3_3_COMBINED.py

# Check what's in dataset
ls -la onyx_dataset_*/
head -20 onyx_dataset_*/dataset_summary.csv
```

### **Data Preparation**
```bash
# Prepare training data
python src/dataset/01_data_preparation.py

# Check prepared data
ls -la preprocessed_data/
cat preprocessed_data/dataset_stats.json
```

### **Model Training**
```bash
# Train baseline (when implemented)
python src/training/train.py --config configs/baseline.yaml

# Resume from checkpoint
python src/training/train.py --resume checkpoints/best_model.pt

# Train with specific settings
python src/training/train.py \
  --d_model 256 \
  --num_layers 6 \
  --batch_size 16 \
  --lr 1e-4 \
  --epochs 100
```

### **Evaluation**
```bash
# Evaluate on validation set
python src/training/evaluate.py --checkpoint checkpoints/best_model.pt

# Generate sample profiles
python src/training/generate.py \
  --origin "Ethiopia" \
  --process "Washed" \
  --flavors "berries,floral,citrus"
```

### **Testing**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_data_preparation.py::test_load_dataset -v
```

### **Git Workflow**
```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/training-pipeline

# Make changes, commit
git add src/training/train.py tests/test_training.py
git commit -m "feat(training): Add baseline training loop with validation"

# Push and create PR
git push origin feature/training-pipeline
# Then create PR on GitHub: feature/training-pipeline → develop

# After merge, clean up
git checkout develop
git pull origin develop
git branch -d feature/training-pipeline
```

---

## 📚 Key Files Reference

### **Data Files**
- `onyx_dataset_builder_v3_3_COMBINED.py` - Web scraper (564 data points, correct metadata)
- `01_data_preparation.py` - Data loader, encoders, train/val split
- `test_results.json` - Example Onyx profile (reference for validation)

### **Model Files**
- `ROASTFORMER_ARCHITECTURE_REFERENCE.py` - Complete transformer code (686 lines)
  - Section 1: Feature encoding (categorical, continuous, flavors)
  - Section 2: Conditioning module
  - Section 3: Positional encodings (sinusoidal, learned, RoPE)
  - Section 4: RoastFormer architecture
  - Section 5: Training utilities
  - Section 6: Usage examples

### **Documentation Files**
- `README.md` - Project overview, installation, usage
- `ARCHITECTURE_QUICK_REFERENCE.md` - Model design summary
- `FEATURE_EXTRACTION_GUIDE.md` - Feature engineering guide
- `Kraiss_Charlee_RoastFormer.pdf` - Capstone proposal (goals, timeline, metrics)

---

## 🎯 Immediate Next Steps for Claude Code

### **Priority 1: Data Validation Pipeline**
**Goal**: Ensure scraped Onyx data is high quality before training

**Tasks**:
1. Create `src/utils/validation.py`
   - Temperature range checks (350-450°F)
   - Profile duration checks (400-1000 points)
   - Feature completeness checks
   - Physics constraint validation

2. Create `tests/test_validation.py`
   - Unit tests for each validation function
   - Test with known good/bad profiles

3. Add validation to data prep pipeline
   - Filter out invalid profiles
   - Report validation statistics

**Success**: All Onyx profiles pass validation, ready for training

### **Priority 2: Training Pipeline**
**Goal**: Implement complete training loop

**Tasks**:
1. Create `src/training/train.py`
   - Load preprocessed data
   - Initialize model
   - Training loop with validation
   - Checkpoint saving
   - Metrics logging

2. Create `src/training/evaluate.py`
   - Load checkpoint
   - Compute validation metrics
   - Generate sample profiles
   - Save results

3. Create `tests/test_training_integration.py`
   - Test training loop runs
   - Test checkpoint saving/loading
   - Test metrics computation

**Success**: Can train baseline model end-to-end

### **Priority 3: Baseline Experiment**
**Goal**: Train first working model

**Tasks**:
1. Train medium model (d_model=256, 6 layers)
2. Validate against Onyx profiles
3. Compute all success metrics
4. Generate example profiles
5. Create visualizations

**Success**: Baseline results documented, ready for ablations

---

## 🔗 External Resources

- **Onyx Coffee Lab**: https://onyxcoffeelab.com
- **Repository**: https://github.com/CKraiss18/roastformer
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **Transformer Tutorial**: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- **DTW Library**: `pip install dtaidistance`

---

## 📝 Notes for Claude Code

### **Project Philosophy**
- **Practical first**: Get working baseline before optimizations
- **Test-driven**: Write tests alongside features
- **Domain-aware**: Validate against coffee roasting physics
- **Iterative**: Small experiments, frequent validation
- **Documented**: Clear commit messages, inline comments

### **Code Style**
- Python 3.8+ (f-strings, type hints)
- PyTorch conventions (snake_case for functions, CamelCase for classes)
- Docstrings (Google style)
- Line length: 100 characters
- Use `black` for formatting (if requested)

### **Working with Small Data**
This is a 28-36 sample dataset. Key strategies:
- Use small models (avoid overfitting)
- Heavy regularization (dropout, weight decay)
- Physics-based constraints (domain knowledge)
- Careful train/val splits (stratified if possible)
- Early stopping (patience=10)
- Visual inspection of every generated profile

### **When Stuck**
1. Check `test_results.json` - reference Onyx profile
2. Review `ROASTFORMER_ARCHITECTURE_REFERENCE.py` - complete model code
3. Check `01_data_preparation.py` - data loading example
4. Refer to proposal PDF - success metrics, timeline
5. Ask for clarification - I'm here to help!

---

## ✅ Checklist Before Starting

- [ ] Repository cloned: `git clone https://github.com/CKraiss18/roastformer.git`
- [ ] Python environment: `python -m venv venv && source venv/bin/activate`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Data collected: `python src/dataset/onyx_dataset_builder_v3_3_COMBINED.py`
- [ ] Tests pass: `pytest tests/ -v` (after creating initial tests)
- [ ] Git configured: `git config user.name` and `user.email` set

---

**Ready to build RoastFormer! ☕🤖**

*Last updated: November 3, 2024*
*For questions: Reference this file, proposal PDF, or architecture reference*
