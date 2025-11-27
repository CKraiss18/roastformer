# RoastFormer Evaluation Checklist

**Purpose**: Step-by-step guide for running evaluation and completing documentation

**Status**: Ready to execute (Tuesday, Nov 19)

---

## 🎯 Pre-Evaluation Setup

### 1. Upload Best Checkpoint to Google Drive
- [ ] Locate: `checkpoints/baseline_sinusoidal/best_model.pt` (from training results)
- [ ] Upload to: `roastformer_data/checkpoints/`
- [ ] Verify file size: Should be ~10-50 MB

### 2. Verify Evaluation Notebook Ready
- [ ] File: `RoastFormer_Evaluation_Demo.ipynb`
- [ ] Already in Google Drive from previous upload
- [ ] No changes needed - template is ready

---

## 📊 Run Evaluation (Colab)

### Cell Execution Order:

**Cells 1-5: Setup & Load**
- [ ] Mount Google Drive
- [ ] Install dependencies
- [ ] Load preprocessed data
- [ ] Load best checkpoint
- [ ] Verify model loaded successfully

**Cells 6-10: Compute Metrics**
- [ ] Run validation set evaluation
- [ ] Compute MAE (Mean Absolute Error in °F)
- [ ] Compute DTW (Dynamic Time Warping distance)
- [ ] Check physics compliance (monotonicity, heating rates)
- [ ] Check finish temperature accuracy

**Cells 11-15: Visualizations**
- [ ] Generate 5 sample profiles
- [ ] Plot real vs. generated comparisons
- [ ] Create error distribution plots
- [ ] Generate attention heatmaps (if applicable)

**Cell 16: Save Results**
- [ ] Download generated profiles
- [ ] Download visualizations
- [ ] Download metrics summary JSON

---

## 📝 Fill Documentation

### File: `docs/EVALUATION_FRAMEWORK.md`

Replace placeholders with actual values:

```markdown
## Validation Set Performance

- **Mean Absolute Error (MAE)**: [X.XX]°F
- **Dynamic Time Warping (DTW)**: [X.XX]
- **Physics Compliance Rate**: [XX]%
- **Finish Temp Accuracy (±10°F)**: [XX]%
```

**What to fill**:
- [ ] Section 3.1: MAE value from Cell 7
- [ ] Section 3.1: DTW value from Cell 8
- [ ] Section 3.2: Physics compliance from Cell 9
- [ ] Section 3.2: Finish temp accuracy from Cell 10
- [ ] Section 4: Add 2-3 example profile comparisons (with images)
- [ ] Section 5: Update limitations based on findings

### Add Training Results Reference

At the end of `EVALUATION_FRAMEWORK.md`, add:

```markdown
## Training Experiments Summary

For detailed training results and ablation studies, see:
- `docs/TRAINING_RESULTS_ANALYSIS.md`

**Winner**: Baseline with Sinusoidal PE (Val Loss: 70,947.55°F)
```

---

## 🎨 Save Visualizations

### Download from Colab:
- [ ] `profile_comparison_1.png` (example 1)
- [ ] `profile_comparison_2.png` (example 2)
- [ ] `profile_comparison_3.png` (example 3)
- [ ] `error_distribution.png`
- [ ] `metrics_summary.png`

### Save locally:
- [ ] Create: `results/visualizations/`
- [ ] Move downloaded images there
- [ ] Reference in `EVALUATION_FRAMEWORK.md`

---

## ✅ Verification Checklist

Before moving to next phase:

- [ ] Evaluation notebook ran successfully (no errors)
- [ ] All metrics computed and saved
- [ ] 5+ sample profiles generated
- [ ] Visualizations downloaded
- [ ] `EVALUATION_FRAMEWORK.md` filled with actual values
- [ ] Training reference added to evaluation doc

---

## 🚨 Expected Results (Sanity Check)

Based on training results, expect:

- **MAE**: ~5-10°F (typical for this task)
- **DTW**: <50 (lower is better)
- **Physics Compliance**: >90% (should be high)
- **Finish Temp Accuracy**: >80% (within ±10°F)

If results are far outside these ranges, investigate:
- Check if correct checkpoint loaded
- Verify data preprocessing matches training
- Review generated profiles visually

---

## 📅 Timeline

**Tuesday Nov 19** (Today):
- Morning: Run evaluation notebook (1 hour)
- Afternoon: Fill EVALUATION_FRAMEWORK.md (1 hour)
- Evening: Review and verify (30 min)

**Wednesday Nov 20**:
- Draft critical analysis using training + evaluation results

---

## 🔗 Files Reference

**Completed**:
- ✅ `docs/TRAINING_RESULTS_ANALYSIS.md` - Training experiments summary
- ✅ `docs/PROJECT_COMPLETION_PLAN.md` - Master roadmap

**To Complete**:
- ⏳ `docs/EVALUATION_FRAMEWORK.md` - Fill after running evaluation
- ⏳ `results/visualizations/` - Save generated images

**Next Phase**:
- 📝 `docs/CRITICAL_ANALYSIS.md` (Nov 20-22)
- 📝 `docs/MODEL_CARD.md` (Nov 24-25)
- 🎤 Presentation slides (Nov 26-28)

---

**Ready to evaluate! 🚀**

*Note: This checklist aligns with PROJECT_COMPLETION_PLAN.md timeline*
