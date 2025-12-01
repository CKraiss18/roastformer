# Colab Training Suite Update Plan

**Date**: November 19, 2025
**Goal**: Update Colab training suite with normalization fix

---

## 📋 Plan Overview

### Phase 1: Train Tiny Model Locally (15-20 min) ✅ Ready to run
### Phase 2: Update Colab Suite (10 min) - If tiny model passes
### Phase 3: Upload & Run on Colab GPU (30-60 min) - Full experiment suite

---

## Phase 1: Local Tiny Model Training

**Command**:
```bash
python train_tiny_model.py
```

**Expected Results** (20 epochs):
- Epoch 10: RMSE ~60°F
- Epoch 20: RMSE ~30-40°F
- Training time: ~15-20 min on CPU

**Success Criteria**:
- ✅ RMSE < 60°F
- ✅ Loss decreases smoothly
- ✅ No early stopping due to stagnation

**If passes** → Move to Phase 2
**If fails** → Debug, then Phase 2 anyway (to show fix on GPU)

---

## Phase 2: Update Colab Training Suite

### Changes Needed

**1. Add Normalized Loader to Colab**

Upload these files to Colab:
```
src/dataset/preprocessed_data_loader_NORMALIZED.py
src/model/transformer_adapter.py (updated generate method)
train_transformer.py (updated import)
```

**2. Create Updated Training Suite Notebook**

`RoastFormer_Training_Suite_FIXED.ipynb`:

```python
# Cell 1: Setup (same as before)

# Cell 2: Install packages
!pip install pandas scikit-learn matplotlib seaborn numpy

# Cell 3: Mount Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Cell 4: Navigate
%cd /content/gdrive/MyDrive/"Colab Notebooks"/"GEN_AI"

# Cell 5: Extract data (same)

# Cell 6: EXPERIMENTS - COMPARISON MODE
EXPERIMENTS = {
    # ═══ FIXED EXPERIMENTS (With Normalization) ═══
    'tiny_normalized': True,          # d=64, should work well
    'micro_normalized': True,         # d=32, fast baseline
    'medium_normalized': True,        # d=128, if time permits

    # ═══ BROKEN EXPERIMENTS (For comparison) ═══
    'tiny_broken': False,             # Same config, no normalization
    # Set to True only if you want to show the broken version
}

# Cell 7: Configuration with normalization flag
def get_config(exp_name, use_normalization=True):
    config = BASE_CONFIG.copy()

    if 'micro' in exp_name:
        config.update({
            'd_model': 32,
            'num_layers': 2,
            'nhead': 2,
            'dim_feedforward': 128,
            ...
        })
    elif 'tiny' in exp_name:
        config.update({
            'd_model': 64,
            'num_layers': 3,
            'nhead': 4,
            'dim_feedforward': 256,
            ...
        })
    elif 'medium' in exp_name:
        config.update({
            'd_model': 128,
            'num_layers': 4,
            'nhead': 4,
            'dim_feedforward': 512,
            ...
        })

    # Toggle normalization
    config['use_normalization'] = use_normalization
    config['data_loader'] = 'NORMALIZED' if use_normalization else 'BROKEN'

    return config

# Cell 8: Training loop with comparison
for exp_name, enabled in EXPERIMENTS.items():
    if not enabled:
        continue

    use_norm = 'normalized' in exp_name
    config = get_config(exp_name, use_normalization=use_norm)

    # Import correct loader
    if use_norm:
        from src.dataset.preprocessed_data_loader_NORMALIZED import PreprocessedDataLoader
    else:
        from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

    # Train...
    trainer = TransformerTrainer(config)
    # ...

# Cell 9: COMPARISON RESULTS TABLE
print("="*80)
print("BEFORE vs AFTER FIX COMPARISON")
print("="*80)

results_df = pd.DataFrame({
    'Experiment': [...],
    'Normalization': [...],
    'Final RMSE (°F)': [...],
    'Epochs to convergence': [...],
    'Generation': [...]
})

print(results_df.to_string())

# Expected output:
# Experiment     | Normalization | Final RMSE | Epochs | Generation
# --------------|---------------|------------|--------|------------
# tiny_broken   | ❌ No        | 274°F      | 16*    | Constant
# tiny_normalized| ✅ Yes       | 40°F       | 15     | Varying
# micro_normalized| ✅ Yes      | 79°F       | 8      | Varying
# medium_normalized| ✅ Yes     | 25°F       | 20     | Smooth

# * Early stopped, not converged

# Cell 10: Teacher Forcing Evaluation (fixed models only)

# Cell 11: Generation Comparison
# Show side-by-side: broken (flat) vs fixed (curves)

# Cell 12: Package results
```

### Key Additions to Suite

**New Cell: "Why the Fix Works"**
```markdown
## Why Temperature Normalization Fixed the Problem

**Before (Broken)**:
- Network receives raw temps: 150-450°F
- Network naturally outputs: ~0-10
- Mismatch: Can't learn 40x scaling needed
- Result: Constant predictions

**After (Fixed)**:
- Network receives normalized temps: 0-1
- Network naturally outputs: 0-1
- Match: Direct correspondence
- Result: Proper learning!

**Analogy**: Imagine teaching someone to count, but you give them problems in feet and expect answers in millimeters. They'll never learn the pattern. Same with neural networks - input and output scales must match.
```

**New Cell: "The Debugging Process"**
```markdown
## How We Found This Bug

1. **Initial Observation**: All models predict constant values
2. **Hypothesis**: Model too large → Test smaller models
3. **Result**: ALL fail identically (5 experiments)
4. **Insight**: Not hyperparameters, must be fundamental
5. **Analysis**: Training logs show only 2.8% improvement
6. **Root Cause**: Scale mismatch - no normalization
7. **Fix**: Normalize temps to [0, 1]
8. **Validation**: 76.9% improvement in 5 epochs!

**Key Lesson**: Systematic debugging > random tuning
```

---

## Phase 3: Run on Colab GPU

### Upload to Colab

```bash
# Zip everything
zip -r roastformer_FIXED_$(date +%Y%m%d_%H%M%S).zip \
    src/ \
    train_transformer.py \
    RoastFormer_Training_Suite_FIXED.ipynb \
    preprocessed_data/ \
    -x "*.pyc" "*__pycache__*"

# Upload to Google Drive
# Then extract in Colab
```

### Expected Runtime

| Experiment | Config | GPU Time | CPU Time |
|------------|--------|----------|----------|
| Micro (d=32) | 2 layers, 8 epochs | ~3 min | ~10 min |
| Tiny (d=64) | 3 layers, 20 epochs | ~10 min | ~30 min |
| Medium (d=128) | 4 layers, 30 epochs | ~25 min | ~90 min |

**Total with 3 experiments**: ~40 min on GPU, ~2.5 hours on CPU

### Success Criteria

**After GPU training, you should see:**

✅ **Normalized models**:
- Micro: RMSE ~79°F (matches local test)
- Tiny: RMSE ~30-50°F
- Medium: RMSE ~20-30°F (if trained)

❌ **Broken model (if included for comparison)**:
- RMSE ~274°F
- Constant predictions
- Early stopping with no learning

**Comparison table shows**:
- 3-10x better RMSE with normalization
- 10-27x faster convergence
- Working generation vs constant

---

## Documentation for Presentation

### Slides to Create

**Slide 1: The Problem**
- "Initial model predicted constant 16°F"
- Show flat line graph

**Slide 2: Recovery Attempts**
- "Tested 5 configurations - all failed"
- Table showing tiny/micro/low LR/high dropout

**Slide 3: The Breakthrough**
- "Training logs revealed: only 2.8% improvement"
- "Models weren't learning at all!"

**Slide 4: Root Cause**
- "Missing temperature normalization"
- Diagram: Raw temps (150-450) vs Network outputs (~0)

**Slide 5: The Fix**
- Code snippet: normalize to [0, 1]
- "One line of code, 27x speedup!"

**Slide 6: Results**
- Comparison table: Before vs After
- Graph: Broken (flat) vs Fixed (curve)

**Slide 7: Lessons**
- Systematic debugging
- Preprocessing is critical
- Scale mismatch invisible but deadly

### Key Numbers to Memorize

- **5 experiments** → all failed → deeper issue
- **2.8% improvement** → not learning
- **27x faster** → after fix
- **76.9% loss reduction** → in just 5 epochs
- **79°F** → micro model RMSE (vs 274°F before)
- **24 hours** → from discovery to working fix

---

## Timeline

**Now**: Run `python train_tiny_model.py` (~20 min)

**If RMSE < 60°F**:
- Update Colab suite with normalization
- Upload to Google Drive
- Run on GPU (~40 min)
- Download results
- Create presentation materials

**If RMSE > 60°F**:
- Still update Colab (to show fix works on GPU)
- Use micro model results (79°F) for now
- Document that tiny needs more tuning

**End result**: Working model + complete debugging narrative for presentation! 🎯

---

Ready to run: `python train_tiny_model.py`
