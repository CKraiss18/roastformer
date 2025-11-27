# RoastFormer Model Card

**Version**: v1.0
**Date**: November 2024
**Authors**: Charlee Kraiss
**Institution**: Vanderbilt University - Generative AI Theory (Fall 2024)

---

## Model Details

### Architecture
- **Type**: Transformer Decoder-Only (Autoregressive)
- **Layers**: 6
- **Hidden Dimension (d_model)**: 256
- **Attention Heads**: 8
- **Feed-Forward Dimension**: 1024 (4x d_model)
- **Total Parameters**: 6,376,673
- **Positional Encoding**: Sinusoidal (best performance vs RoPE/Learned)

### Novel Contribution
**Flavor-Conditioned Generation**: First transformer model to condition coffee roast profile generation on desired flavor outcomes (e.g., "berries", "chocolate", "floral"). Validated with 14% performance improvement over no-flavor baseline.

### Training Configuration
```python
Batch Size: 16
Learning Rate: 1e-4
Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.01)
Scheduler: CosineAnnealingLR (T_max=100)
Loss Function: MSE (Mean Squared Error)
Dropout: 0.1
Gradient Clipping: 1.0
Early Stopping: Patience=20
```

### Input Features (17 Total)

**Categorical Features (5)**:
- Origin (20 unique: Ethiopia, Colombia, Guatemala, etc.)
- Process Method (6 unique: Washed, Natural, Honey, Anaerobic, etc.)
- Bean Variety (15 unique: Heirloom, Caturra, Bourbon, etc.)
- Roast Level (4 unique: Expressive Light, Balanced, Medium, Dark)
- Flavor Notes (40 unique, multi-hot encoded: berries, chocolate, floral, citrus, etc.)

**Continuous Features (4)**:
- Target Finish Temperature (390-430°F, normalized)
- Altitude (1000-2300 MASL, normalized)
- Bean Density Proxy (normalized by origin)
- Caffeine Content (normalized by variety)

### Output
- **Sequence**: Temperature values at 1-second intervals
- **Length**: Variable (400-1000 time steps, ~7-16 minutes)
- **Range**: 250-450°F (constrained during generation)
- **Format**: Autoregressive generation (predicts next temperature given previous)

---

## Intended Use

### Primary Use Case
**Roast Profile Generation for Specialty Coffee**: Generate data-driven starting roast profiles for new coffees based on bean characteristics and desired flavor outcomes. Reduces experimentation time from 10-20 trial roasts to 2-3 refinements.

### Target Users
- Specialty coffee roasters introducing new coffees
- Coffee quality control labs
- Roasting research and education
- Roast profile database generation

### Out-of-Scope Uses
- **Not for production roasting without validation**: Current physics compliance challenges (0%) require human verification
- **Not for commodity coffee**: Trained on specialty-grade (80+ score) profiles only
- **Not for roasters outside 10-50 lb batch range**: Trained on Loring S70 data

---

## Training Data

### Dataset: Onyx Coffee Lab Validation Set
- **Source**: https://onyxcoffeelab.com (2019 US Roaster Champions)
- **Size**: 144 roast profiles
  - Training: 123 profiles (85%)
  - Validation: 21 profiles (15%)
- **Roaster**: Loring S70 Peregrine
- **Batch Size**: 10-50 lbs
- **Temporal Coverage**: 2019-2024
- **Geographic Coverage**: 15+ coffee origins
- **Resolution**: 1-second intervals

### Data Characteristics
- **Profile Duration**: 7-16 minutes (mean: 11.2 min)
- **Charge Temperature**: 415-435°F (mean: 426°F)
- **Drop Temperature**: 390-425°F (mean: 405°F)
- **Roast Levels**: Light (72%), Medium (23%), Dark (5%)
- **Completeness**: 100% temperature data, 95% altitude, 100% flavors

### Known Limitations
1. **Single roaster**: All data from Onyx (Loring S70), limits generalization
2. **Small sample size**: 144 profiles - exposure bias amplified
3. **Light roast bias**: 72% light roasts, underrepresents dark
4. **Specialty only**: No commodity-grade coffees (scoring <80)
5. **Modern style**: High-charge, fast development (not traditional)

---

## Performance

### Training Performance (Best Model: d=256)

**Validation Metrics**:
- **RMSE**: 10.4°F ✅ (lowest among all model sizes)
- **MAE**: 8.2°F
- **Finish Temperature Accuracy**: 95% within ±10°F
- **Convergence**: 42 epochs (early stopping at epoch 62)

**Surprising Finding**: Largest model (d=256, 6.4M parameters) outperformed smaller variants despite 51,843:1 parameter-to-sample ratio. Key insight: **normalization + regularization > capacity limits**. With proper weight decay, dropout, and early stopping, larger models leverage capacity to learn complex roast dynamics.

### Ablation Study Results

| Experiment | RMSE | Finding |
|-----------|------|---------|
| **Model Size** | | |
| d=32 | 43.8°F | Too small, underfits |
| d=64 | 23.4°F | Solid baseline |
| d=128 | 16.5°F | Strong performance |
| **d=256** | **10.4°F** | **Best** ✅ |
| **Positional Encoding** | | |
| **Sinusoidal** | **23.4°F** | **Best** ✅ |
| RoPE | 28.1°F | Worse despite complexity |
| Learned | 43.8°F | Overfits small data |
| **Flavor Conditioning** | | |
| Without flavors | 27.2°F | Baseline |
| **With flavors** | **23.4°F** | **+14% improvement** ✅ |

**Key Validation**: Flavor conditioning provides significant performance gain, validating the novel contribution.

### Generation Performance (Autoregressive Evaluation)

**Unconstrained Generation**:
- **MAE**: 25.3°F (reasonable temperature accuracy)
- **RMSE**: 29.8°F
- **Finish Temperature**: 50% within ±10°F
- **Physics Compliance**: 0% ❌ (monotonicity violations, unbounded heating rates)

**Challenge Identified**: **Autoregressive Exposure Bias**
- Training: Model sees real previous temperatures (teacher forcing) → learns patterns ✅
- Generation: Model sees own predictions → errors compound ❌
- Gap: 10.4°F RMSE (training) vs 25.3°F MAE (generation)

**Attempted Solution - Physics-Constrained Generation**: FAILED
- Enforced monotonicity, bounded RoR (20-100°F/min), smooth transitions
- Result: MAE 25→114°F (4.5x worse), generated linear ramps instead of curves
- Lesson: **Post-processing constraints cannot fix training issues**

---

## Limitations & Risks

### Technical Limitations

1. **Exposure Bias** (Critical)
   - **Issue**: Model trained with teacher forcing but generates from own predictions
   - **Impact**: 0% physics compliance, unrealistic profiles during autoregressive generation
   - **Mitigation**: Requires scheduled sampling or physics-informed losses (training-time fixes)

2. **Physics Compliance** (Critical)
   - **Issue**: Generated profiles violate roasting physics (non-monotonic, unbounded RoR)
   - **Impact**: Profiles unusable without manual validation
   - **Mitigation**: Human verification required before any practical use

3. **Single-Roaster Bias**
   - **Issue**: All training data from Onyx (Loring S70, modern light roast style)
   - **Impact**: May not generalize to other roasters, machines, or styles
   - **Mitigation**: Validate on user's equipment before trusting outputs

4. **Small Dataset**
   - **Issue**: Only 144 profiles amplifies overfitting and exposure bias
   - **Impact**: Limited pattern diversity, potential memorization
   - **Mitigation**: Heavy regularization applied, but scaling to 500+ profiles recommended

### Risks & Ethical Considerations

1. **Production Use Risk** ⚠️
   - **Risk**: Using unvalidated profiles could damage expensive equipment or beans
   - **Severity**: HIGH (equipment damage $50k+, batch loss $200+)
   - **Mitigation**: NEVER use generated profiles without expert validation

2. **Overreliance on Model**
   - **Risk**: Roasters might trust model over domain expertise
   - **Severity**: MEDIUM (suboptimal roasts, inconsistent quality)
   - **Mitigation**: Position as "starting point" tool, not replacement for expertise

3. **Data Representativeness**
   - **Risk**: Model trained on championship-level roaster, may not reflect typical practices
   - **Severity**: MEDIUM (unrealistic expectations)
   - **Mitigation**: Clearly document training data source and limitations

4. **Intellectual Property**
   - **Risk**: Onyx roast profiles are proprietary trade secrets
   - **Severity**: LOW (public data used with attribution)
   - **Mitigation**: All data sourced from public website, no private profiles

---

## Future Improvements

### Immediate (Technical Fixes)

1. **Scheduled Sampling** (Bengio et al., 2015)
   - Gradually transition from teacher forcing to model predictions during training
   - Addresses exposure bias at the source
   - Expected impact: Reduce generation MAE 25→15°F, improve physics compliance

2. **Physics-Informed Loss Functions**
   - Add penalty terms for monotonicity violations, unbounded RoR
   - Model learns to respect constraints during training
   - Expected impact: 0% → 80%+ physics compliance

3. **Non-Autoregressive Generation**
   - Explore diffusion models for profile generation
   - Generate entire sequence at once (no error accumulation)
   - Expected impact: Eliminate exposure bias entirely

### Long-Term (Scalability)

4. **Multi-Roaster Dataset** (500+ profiles)
   - Include diverse roasters, machines, styles
   - Transfer learning across roaster styles
   - Expected impact: Better generalization, fewer artifacts

5. **Multi-Modal Conditioning**
   - Add roaster type, ambient conditions, bean moisture
   - Richer context for generation
   - Expected impact: +10-20% accuracy improvement

6. **Real-Time Feedback Loop**
   - Fine-tune on user's successful roasts
   - Personalized profile generation
   - Expected impact: Adapt to specific equipment and preferences

---

## Course Connections (Generative AI Theory)

**Week 2: Neural Network Fundamentals**
- **Applied**: Temperature normalization (critical bug fix)
- **Impact**: 27x faster convergence
- **Lesson**: "Networks output values near initialization scale (0-10). We asked for raw temps (150-450°F). Normalization was THE critical fix."

**Week 4: Autoregressive Modeling & Exposure Bias**
- **Applied**: Sequential temperature generation with teacher forcing
- **Challenge**: Exposure bias (training gap vs generation gap)
- **Lesson**: "Training with real sequences doesn't prepare model for generating from own predictions."

**Week 5: Transformer Architecture & Positional Encodings**
- **Applied**: Compared sinusoidal, RoPE, learned positional encodings
- **Finding**: Sinusoidal best (23.4°F) vs RoPE (28.1°F) - classic methods win on small data
- **Lesson**: "Simpler methods beat complex ones in low-data regimes."

**Week 6-7: Conditional Generation (Multi-Modal Features)**
- **Applied**: Flavor-conditioned generation (novel contribution)
- **Result**: +14% improvement validates approach
- **Lesson**: "Task-relevant conditioning (flavors) improves generation quality measurably."

**Week 8: Small-Data Regime Strategies**
- **Applied**: Heavy regularization (dropout=0.1, weight_decay=0.01, early stopping)
- **Surprising Result**: d=256 with 51,843:1 ratio achieved BEST performance
- **Lesson**: "Normalization + regularization > capacity limits. Being experimentally wrong taught more than being theoretically correct."

**Week 9: Evaluation Methodology & Domain-Specific Metrics**
- **Applied**: Physics-based validation (monotonicity, bounded RoR, smoothness)
- **Finding**: Standard metrics (RMSE) don't capture domain constraints
- **Lesson**: "Generic metrics mislead. Domain-specific validation reveals real limitations."

---

## Model Access & Reproducibility

### Checkpoint
- **Location**: `checkpoints/best_model_d256_epoch42.pt`
- **Size**: 24.3 MB
- **SHA256**: (compute hash before distribution)

### Code Repository
- **GitHub**: https://github.com/CKraiss18/roastformer
- **Branch**: `main`
- **Commit**: (add commit hash for reproducibility)

### Dependencies
```
python >= 3.8
torch >= 2.0.0
numpy >= 1.23.0
pandas >= 1.5.0
```

### Training Reproducibility
```bash
# Reproduce training
python train_transformer.py \
  --d_model 256 \
  --num_layers 6 \
  --num_heads 8 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --seed 42
```

### Generation Example
```python
from src.model.transformer_adapter import TransformerAdapter

# Load model
model = TransformerAdapter.from_pretrained('checkpoints/best_model_d256_epoch42.pt')

# Generate profile
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

# Validate before use!
assert validate_physics(profile), "Profile failed physics checks - DO NOT USE"
```

---

## Citation

If you use RoastFormer in your work, please cite:

```bibtex
@software{kraiss2024roastformer,
  author = {Kraiss, Charlee},
  title = {RoastFormer: Flavor-Conditioned Coffee Roast Profile Generation with Transformers},
  year = {2024},
  institution = {Vanderbilt University},
  course = {Generative AI Theory (Fall 2024)},
  url = {https://github.com/CKraiss18/roastformer}
}
```

---

## Contact & Support

**Author**: Charlee Kraiss
**Email**: charlee.kraiss@vanderbilt.edu
**GitHub Issues**: https://github.com/CKraiss18/roastformer/issues

**Acknowledgments**: Onyx Coffee Lab for publicly sharing roast profile data. Course instructor and TAs for guidance on transformer implementation and evaluation methodology.

---

**Last Updated**: November 20, 2024
**Status**: Research Prototype (NOT production-ready)
**License**: MIT (code), CC BY-NC 4.0 (data/documentation)
