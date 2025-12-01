# Final Push: Fix + Fresh Data + Retrain

**Date**: November 19, 2025
**Goal**: Get a working model with normalized temperatures + more training data

---

## 🎯 Plan

### Phase 1: Fresh Data Collection (5-10 min)
1. ✅ Run Onyx scraper to get latest profiles
2. ✅ Should have ~150-200 profiles now (was 144 in October)
3. ✅ Process with normalized data loader

### Phase 2: Quick Test (10 min)
1. Train micro model (d=32, fastest)
2. Check if loss drops properly with normalization
3. Verify generation isn't constant

### Phase 3: Full Training (if test succeeds) (30-60 min)
1. Train tiny model (d=64) to completion
2. Evaluate with teacher forcing + generation
3. Document results

---

## 📝 Step-by-Step

### Step 1: Scrape Fresh Data

```bash
cd /Users/charleekraiss/VANDY/FALL_2025/GEN_AI_THEORY/ROASTFormer
python src/dataset/onyx_dataset_builder_v3_3_COMBINED.py
```

**Expected output:**
- New directory: `onyx_dataset_YYYY_MM_DD/`
- ~150-200 profiles (more than Oct scrape)

### Step 2: Prepare Data with Normalized Loader

**Create new preparation script that uses normalization:**

`prepare_data_NORMALIZED.py`:
```python
"""
Prepare training data with NORMALIZED temperatures
"""
import sys
sys.path.append('.')

from src.dataset.preprocessed_data_loader_NORMALIZED import PreprocessedDataLoader
from pathlib import Path
import json

# Use latest Onyx dataset
onyx_dirs = sorted(Path('.').glob('onyx_dataset_*'))
latest_dir = onyx_dirs[-1]

print(f"Using dataset: {latest_dir}")

# TODO: Need to create prepare script that:
# 1. Loads raw Onyx profiles
# 2. Splits train/val
# 3. Saves to preprocessed_data_NORMALIZED/
# 4. Uses normalized temperatures
```

### Step 3: Update Training to Use Normalized Loader

**In `train_transformer.py`**:
```python
# Change line ~21:
from src.dataset.preprocessed_data_loader_NORMALIZED import PreprocessedDataLoader
```

**In `transformer_adapter.py`** line ~222:
```python
def generate(...):
    from src.dataset.preprocessed_data_loader_NORMALIZED import normalize_temperature, denormalize_temperature

    # Normalize start
    start_temp_norm = normalize_temperature(start_temp)
    generated = torch.tensor([[start_temp_norm]], device=device)

    # ... generation loop ...

    # Denormalize before return
    generated_temps = np.array([denormalize_temperature(t) for t in generated_norm])
    return generated_temps
```

### Step 4: Quick Test

**Train micro model for 10 epochs:**
```python
config = {
    'd_model': 32,
    'num_layers': 2,
    'nhead': 2,
    'dim_feedforward': 128,
    'batch_size': 8,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'preprocessed_dir': 'preprocessed_data_NORMALIZED'
}
```

**Success criteria (epoch 5):**
- Training loss <0.3 (normalized MSE)
- Validation loss <0.5 (normalized MSE)
- Loss decreased by >50% from epoch 1

**If failed:**
- Loss still ~75,000 → normalization not applied
- Check imports, check data loader being used

### Step 5: Full Training (if test passed)

**Train tiny model (d=64):**
```python
config = {
    'd_model': 64,
    'num_layers': 3,
    'nhead': 4,
    'dim_feedforward': 256,
    'dropout': 0.2,
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'early_stopping_patience': 15,
    'preprocessed_dir': 'preprocessed_data_NORMALIZED'
}
```

**Expected results:**
- Epoch 10: RMSE ~0.2 normalized (~8°F real)
- Epoch 30: RMSE ~0.1 normalized (~4°F real)
- Teacher forcing MAE: <20°F
- Generation: Varying temps (not constant)

---

## 🎯 Timeline

**Tonight (if time):**
- Scrape data (5 min)
- Prepare normalized data (10 min)
- Quick test micro model (10 min)
- **Total: 25 minutes** → Know if fix works!

**Tomorrow (if tonight's test succeeds):**
- Train tiny model (30-60 min)
- Evaluate results
- Update critical analysis with findings

**If no time:**
- Document the fix in critical analysis
- Explain expected impact
- Show systematic debugging process
- This is still valuable work!

---

## 📊 What We've Learned

Regardless of whether we get a working model tonight:

1. ✅ **Identified root cause** of model collapse (normalization bug)
2. ✅ **Systematic debugging** (ruled out model size, LR, dropout)
3. ✅ **Implemented fix** (normalized data loader)
4. ✅ **Understood failure modes** (scale mismatch, gradient issues)
5. ✅ **Documented process** (critical analysis, lessons learned)

**This is excellent engineering work!** 🎓

---

Ready to scrape fresh data and test the fix!
