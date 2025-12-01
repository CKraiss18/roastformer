# Google Colab Training Guide for RoastFormer

**Complete workflow for training on Colab and sharing results with Claude**

Date: November 2025
Author: Charlee Kraiss

---

## 🎯 Why Use Colab?

| Aspect | Your Mac (CPU) | Colab (Free GPU) | Winner |
|--------|---------------|------------------|--------|
| **Speed** | 3-6 hours | 30-60 min | ✅ Colab |
| **Cost** | Free | Free | 🤝 Tie |
| **Laptop Battery** | Drains battery | No impact | ✅ Colab |
| **Can use Mac for other work** | No | Yes | ✅ Colab |
| **Setup complexity** | None (works now) | 5 min one-time | 🤝 Close |

**Recommendation:** Use Colab for training, Mac for evaluation/generation.

---

## 📋 Complete Workflow

### **Step 1: Prepare Data for Upload (On Your Mac)**

```bash
cd ~/VANDY/FALL_2025/GEN_AI_THEORY/ROASTFormer

# Package everything for Colab
python package_for_colab.py
```

**Output:** `roastformer_data_YYYYMMDD_HHMMSS.zip` (~10-20 MB)

**What's included:**
- Preprocessed training data (86 profiles)
- Preprocessed validation data (15 profiles)
- All source code (data loaders, model, training script)
- Dataset statistics

---

### **Step 2: Upload to Colab**

1. **Go to Colab:**
   - Visit: https://colab.research.google.com
   - Sign in with your Google account

2. **Upload the notebook:**
   - Click **File → Upload notebook**
   - Select: `RoastFormer_Colab_Training.ipynb`

3. **Enable GPU:**
   - Click **Runtime → Change runtime type**
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free tier)
   - Click **Save**

4. **Verify GPU:**
   - Run the first cell (GPU check)
   - Should see: `✅ GPU ready for training!`
   - GPU: Tesla T4 (or similar)

---

### **Step 3: Upload Your Data**

When you reach Cell 2 in the notebook:

1. Click the **Choose Files** button
2. Select your `roastformer_data_*.zip` file
3. Wait for upload to complete (1-2 minutes)
4. Data will be automatically extracted

**Verify:**
- Cell 3 will show all files are present
- Dataset stats will be displayed

---

### **Step 4: Configure Training**

**Cell 4** contains the training configuration.

**For Baseline Training (Recommended):**
```python
config = {
    'd_model': 256,              # Medium model
    'nhead': 8,
    'num_layers': 6,
    'positional_encoding': 'sinusoidal',
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    # ... (defaults are good)
}
```

**For Quick Test (5-10 min):**
```python
config = {
    'd_model': 128,              # Small model
    'num_layers': 4,
    'num_epochs': 20,            # Fewer epochs
    # ... rest as default
}
```

**For Ablation Study - Learned Positional Encoding:**
```python
config = {
    'd_model': 256,
    'positional_encoding': 'learned',  # ← Change this
    # ... rest same as baseline
}
```

---

### **Step 5: Train the Model**

**Run Cell 5** to start training.

**What you'll see:**
```
================================================================================
STARTING TRAINING
================================================================================
Device: cuda
Epochs: 100
================================================================================

Loading data...
✓ Loaded 86 training profiles
✓ Loaded 15 validation profiles

Initializing model...
✓ Model initialized: 10,234,567 parameters

Training...
Epoch 1/100
--------------------------------------------------------------------------------
  Train Loss: 1234.5678
  Val Loss:   1456.7890
  LR:         0.000100
  Time:       12.3s

Epoch 2/100
...
```

**Monitor:**
- Loss should decrease over time
- Val loss should roughly follow train loss
- If val loss >> train loss consistently → overfitting

**Typical Training Time:**
- Small model (d_model=128): 15-30 min
- Medium model (d_model=256): 30-60 min
- Large model (d_model=512): 1-2 hours

**💡 Pro Tip:** Colab free tier has session limits (~12 hours). Your training will complete in < 2 hours.

---

### **Step 6: Review Results**

**Cell 6** generates a summary and plots.

**You'll see:**
- Final metrics (best validation loss, final losses)
- Training curves plot
- Model configuration summary

**Good signs:**
- Both train and val loss decrease
- Val loss stabilizes (not increasing)
- Best val loss < 5°F (target from CLAUDE.md)

**Warning signs:**
- Val loss much higher than train → overfitting
- Both losses plateau early → underfitting
- Losses diverge (go to infinity) → learning rate too high

---

### **Step 7: Download Results**

**Cell 7 & 8** package and download results.

**Downloaded file:** `roastformer_results_YYYYMMDD_HHMMSS.zip`

**Contents:**
- `best_transformer_model.pt` - Trained model checkpoint (~10-40 MB)
- `transformer_training_results.json` - Complete metrics
- `training_curves.png` - Visualization
- `training_summary.txt` - Human-readable summary

**Save this zip file!** You'll need it for evaluation and sharing with Claude.

---

### **Step 8: Share Results with Claude**

**On your Mac:**

```bash
cd ~/VANDY/FALL_2025/GEN_AI_THEORY/ROASTFormer

# Extract the results
unzip roastformer_results_*.zip

# Move checkpoint to checkpoints folder
mv best_transformer_model.pt checkpoints/

# Generate Claude-friendly summary
python share_results_with_claude.py
```

**This creates:** `results/claude_summary_YYYYMMDD_HHMMSS.md`

**Open the file and copy/paste into your Claude conversation!**

---

## 🔄 Running Multiple Experiments

### **Ablation Study Workflow**

**Experiment 1: Baseline (Sinusoidal Positional Encoding)**
```python
# Cell 4
config = {
    'd_model': 256,
    'positional_encoding': 'sinusoidal',
    # ...
}
```
- Run training
- Download results → `results_baseline_sinusoidal.zip`

**Experiment 2: Learned Positional Encoding**
```python
# Cell 4
config = {
    'd_model': 256,
    'positional_encoding': 'learned',
    # ...
}
```
- Run training
- Download results → `results_baseline_learned.zip`

**Experiment 3: Different Model Size**
```python
# Cell 4
config = {
    'd_model': 512,  # Large model
    'num_layers': 8,
    # ...
}
```
- Run training
- Download results → `results_large_model.zip`

**Compare:**
- Best validation loss
- Training curves
- Overfitting behavior
- Training time

---

## 📊 Interpreting Results

### **Good Training**

```
Epoch 1/100:   Train: 1234.5°F  Val: 1456.7°F
Epoch 10/100:  Train: 234.5°F   Val: 287.3°F
Epoch 50/100:  Train: 45.2°F    Val: 52.8°F
Epoch 100/100: Train: 12.3°F    Val: 15.6°F
Best val loss: 14.2°F (epoch 95)
```

✅ **Signs:**
- Both decrease smoothly
- Val slightly higher than train (normal)
- Best val loss < 5-10°F
- Early stopping around epoch 95

### **Overfitting**

```
Epoch 1/100:   Train: 1234.5°F  Val: 1456.7°F
Epoch 50/100:  Train: 5.2°F     Val: 125.8°F    ← Val much higher
Epoch 100/100: Train: 0.3°F     Val: 234.6°F    ← Getting worse
```

⚠️ **Signs:**
- Train loss very low, val loss high
- Val loss increasing while train decreases
- Big gap between train and val

**Solutions:**
- Increase dropout (0.1 → 0.2)
- Increase weight decay (0.01 → 0.05)
- Use smaller model
- Enable early stopping

### **Underfitting**

```
Epoch 1/100:   Train: 1234.5°F  Val: 1456.7°F
Epoch 50/100:  Train: 523.4°F   Val: 612.8°F
Epoch 100/100: Train: 487.2°F   Val: 556.3°F    ← Still high
```

⚠️ **Signs:**
- Both losses plateau high
- Little improvement after many epochs
- Losses don't reach < 50°F

**Solutions:**
- Use larger model
- More layers/heads
- Lower learning rate
- Train longer

---

## 💾 Managing Colab Sessions

### **Session Limits**

**Free Tier:**
- Max runtime: ~12 hours
- Can be disconnected after 90 min idle
- GPU access subject to availability

**Tips:**
- Training completes in < 2 hours, so you're safe
- Keep browser tab open
- Check progress periodically

### **If Disconnected**

Colab saves outputs in `/content/` but it's wiped after disconnect.

**Always download results immediately after training!**

If you get disconnected:
- Re-upload notebook
- Re-upload data
- Re-run training (unfortunately)

**Pro Tier ($10/month):**
- Longer runtimes
- Better GPUs (V100/A100)
- Faster training (15-30 min instead of 30-60 min)
- Background execution

---

## 🔍 Troubleshooting

### **"No GPU detected"**

**Fix:**
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save
4. Run GPU check cell again

### **"Cannot allocate memory"**

Model too large for T4 GPU.

**Fix:**
- Reduce `d_model`: 512 → 256
- Reduce `batch_size`: 8 → 4
- Reduce `max_sequence_length`: 800 → 600

### **"Loss is NaN"**

Learning rate too high or numerical instability.

**Fix:**
- Lower `learning_rate`: 1e-4 → 5e-5
- Increase `grad_clip`: 1.0 → 0.5
- Check data for corrupted values

### **Training very slow**

GPU not being used.

**Check:**
- Cell 1 shows GPU is available
- `config['device']` is 'cuda' not 'cpu'
- Restart runtime and try again

---

## 📁 File Organization

**After downloading results:**

```
ROASTFormer/
├── checkpoints/
│   └── best_transformer_model.pt          ← Move here
│
├── results/
│   ├── transformer_training_results.json  ← Move here
│   ├── training_curves.png                ← Move here
│   ├── training_summary.txt               ← Move here
│   └── claude_summary_*.md                ← Generated by share script
│
└── colab_experiments/                     ← Create this
    ├── experiment_1_baseline/
    │   ├── best_transformer_model.pt
    │   └── results.json
    ├── experiment_2_learned_pos/
    │   ├── best_transformer_model.pt
    │   └── results.json
    └── experiment_3_large_model/
        ├── best_transformer_model.pt
        └── results.json
```

---

## 🎓 For Your Capstone

### **Week 2 Plan (This Week)**

**Monday-Tuesday:**
1. Train baseline model on Colab (1 hour)
2. Share results with Claude
3. Evaluate model on Mac

**Wednesday-Thursday:**
1. Ablation study #1: Learned vs Sinusoidal positional encoding (2 hours)
2. Ablation study #2: Model sizes (small/medium/large) (3 hours)
3. Compare results

**Friday:**
1. Select best configuration
2. Final training run
3. Comprehensive evaluation

### **Deliverables**

From Colab training:
- ✅ Trained baseline model
- ✅ Ablation study results
- ✅ Training curves for all experiments
- ✅ Comparison table

From Mac evaluation:
- ✅ Generated sample profiles
- ✅ Physics validation results
- ✅ Real vs generated comparisons

---

## 📞 Getting Help from Claude

**Always share:**
1. Training summary (from `share_results_with_claude.py`)
2. Specific question or concern
3. What you've tried

**Example:**
```
Hey Claude! Just finished training the baseline transformer on Colab.
Here are the results:

[paste training_summary.txt contents]

Question: The validation loss is 4.2°F which seems good, but I'm seeing
the val loss start to increase slightly in the last few epochs. Is this
overfitting? Should I enable early stopping for the next run?
```

Claude will analyze your specific results and give tailored advice!

---

## ✅ Quick Checklist

**Before starting:**
- [ ] Mac has preprocessed data ready
- [ ] `package_for_colab.py` run successfully
- [ ] Colab account ready
- [ ] GPU runtime enabled

**During training:**
- [ ] Data uploaded and extracted
- [ ] Configuration set correctly
- [ ] GPU being used (check cell 1)
- [ ] Monitoring progress

**After training:**
- [ ] Results downloaded
- [ ] Results extracted and organized
- [ ] Summary shared with Claude
- [ ] Checkpoint saved

**For evaluation:**
- [ ] Run `evaluate_transformer.py` on Mac
- [ ] Generate sample profiles
- [ ] Check physics constraints
- [ ] Document findings

---

**Ready to train! 🚀☕**

Questions? Run `python share_results_with_claude.py` and paste the output to get help!
