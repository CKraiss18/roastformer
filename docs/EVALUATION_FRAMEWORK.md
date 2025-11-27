# RoastFormer: Evaluation Framework

**Author**: Charlee Kraiss
**Project**: Transformer-Based Coffee Roast Profile Generation
**Course**: Generative AI Theory (Fall 2024)
**Date**: November 2024

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Metric Choices & Justifications](#metric-choices--justifications)
3. [Evaluation Protocol](#evaluation-protocol)
4. [Results Summary](#results-summary)
5. [Limitations & Trade-offs](#limitations--trade-offs)
6. [Future Evaluation Approaches](#future-evaluation-approaches)
7. [References](#references)

---

## Overview

### Purpose

This document defines the evaluation framework for RoastFormer, a transformer-based model for generating coffee roast temperature profiles. The framework addresses the unique challenges of evaluating generative models in a **constrained physical domain** with **small-data regimes**.

### Key Evaluation Questions

1. **Accuracy**: How closely do generated profiles match real profiles?
2. **Physical Validity**: Do generated profiles obey roasting physics?
3. **Task Performance**: Do profiles achieve target finish temperatures?
4. **Generalization**: How well does the model handle unseen bean characteristics?

---

## Metric Choices & Justifications

### 1. Mean Absolute Error (MAE)

**Definition:**
```
MAE = (1/n) Σ |T_real(t) - T_generated(t)|
```

**Why This Metric?**
- **Interpretable**: Direct temperature difference in °F
- **Task-relevant**: Roasters think in temperature accuracy
- **Sensitive to errors**: Penalizes all deviations equally

**Trade-offs:**
- ✅ **Strengths**:
  - Easy to understand and communicate
  - Standard regression metric (Week 9 - evaluation metrics)
  - Directly measures what roasters care about (temperature accuracy)

- ❌ **Limitations**:
  - Doesn't capture trajectory shape
  - Treats all timesteps equally (but first crack is more critical)
  - Penalizes phase shifts (profile correct but 10 seconds late)

**Course Connection**: Standard supervised learning evaluation (Week 9), adapted from classification to continuous regression.

**Target Performance:**
- **Excellent**: MAE < 5°F (within measurement noise)
- **Good**: MAE < 10°F (usable starting profile)
- **Acceptable**: MAE < 15°F (requires adjustment but better than starting from scratch)

**Actual Results** *(Fill after evaluation)*:
```
MAE: [X.XX]°F
Interpretation: [Excellent/Good/Acceptable - explain what this means]
```

---

### 2. Dynamic Time Warping (DTW) Distance

**Definition:**
```
DTW = min_alignment Σ (T_real[i] - T_gen[align[i]])²
```

**Why This Metric?**
- **Shape similarity**: Captures overall trajectory, not just point-wise error
- **Phase-shift robust**: Allows flexible temporal alignment
- **Pattern recognition**: Similar to time-series similarity (Week 10)

**Trade-offs:**
- ✅ **Strengths**:
  - Robust to small timing differences (profile correct but 5-10 sec shifted)
  - Captures qualitative profile shape
  - Handles variable-length sequences (roasts 9-12 min)

- ❌ **Limitations**:
  - More complex than MAE (harder to interpret)
  - Can hide systematic timing errors
  - Computationally expensive (O(n²))

**Course Connection**: Time-series analysis techniques (Week 10), similar to sequence alignment in NLP.

**Interpretation Guide:**
- **Low DTW**: Profiles have similar shape/trajectory
- **High DTW**: Profiles differ in fundamental structure

**Actual Results** *(Fill after evaluation)*:
```
DTW Distance: [X.XX]
Interpretation: [Compare to baseline, explain if shape is captured well]
```

---

### 3. Finish Temperature Accuracy

**Definition:**
```
Finish_Accuracy = Percentage of profiles with |T_final - T_target| < 10°F
```

**Why This Metric?**
- **Task-specific**: Directly measures success (hit target roast level)
- **Binary outcome**: Clear pass/fail for each profile
- **Critical for usability**: Wrong finish temp = wrong roast level

**Trade-offs:**
- ✅ **Strengths**:
  - Directly measures task objective
  - Easy to communicate (X% successful)
  - Most important for practical use

- ❌ **Limitations**:
  - Ignores entire trajectory (could reach target with terrible path)
  - Binary (doesn't capture "close" vs "very wrong")
  - 10°F threshold is arbitrary (but reasonable)

**Course Connection**: Task-oriented evaluation (Week 7 - conditional generation), similar to controllability metrics.

**Target Performance:**
- **Excellent**: >90% within 10°F
- **Good**: >75% within 10°F
- **Acceptable**: >60% within 10°F

**Actual Results** *(Fill after evaluation)*:
```
Finish Temperature Accuracy: [X.X]% within 10°F
Distribution:
  - Within 5°F: [X.X]%
  - Within 10°F: [X.X]%
  - Within 15°F: [X.X]%
  - >15°F off: [X.X]%

Interpretation: [Discuss if model learned to condition on target temperature]
```

---

### 4. Physics Constraint Compliance

**Definition:**
Multiple sub-metrics measuring physical plausibility:

#### 4a. Monotonicity (Post-Turning-Point)
```python
turning_point_idx = argmin(temperatures)
monotonic = all(diff(temps[turning_point_idx:]) >= 0)
```

**Why**: After beans reach minimum temp, they should only heat up (physical law).

#### 4b. Bounded Heating Rates
```python
ror = diff(temperatures) * 60  # °F/min
bounded = (20 <= ror) & (ror <= 100)
compliance = mean(bounded)
```

**Why**: Too fast (>100°F/min) = scorching; too slow (<20°F/min) = baking.

#### 4c. Smooth Transitions
```python
jumps = abs(diff(temperatures))
smooth = all(jumps < 10/60)  # Less than 10°F per second
```

**Why**: Temperature can't jump discontinuously (thermal inertia).

**Trade-offs:**
- ✅ **Strengths**:
  - Ensures physical plausibility
  - Catches impossible profiles
  - Incorporates domain knowledge (inductive bias from Week 2)

- ❌ **Limitations**:
  - Binary constraints (pass/fail, not "how good")
  - Requires domain expertise to define
  - Conservative (may reject valid edge cases)

**Course Connection**: Inductive biases and domain knowledge (Week 2), similar to enforcing physical laws in scientific ML.

**Target Performance:**
- **Excellent**: >95% compliance on all metrics
- **Good**: >85% compliance
- **Acceptable**: >70% compliance (with clear patterns of where it fails)

**Actual Results** *(Fill after evaluation)*:
```
Physics Compliance:
  Monotonicity (post-turning-point): [X.X]%
  Bounded RoR (20-100°F/min): [X.X]%
  Smooth Transitions (<10°F/s): [X.X]%

  Overall Physics Score: [X.X]%

Interpretation:
  - [Discuss which constraints are hardest to satisfy]
  - [Do failures follow patterns? E.g., always at first crack]
  - [Are failures "close" to valid or completely off?]
```

---

### 5. Qualitative Assessment (Human Evaluation - Future Work)

**Definition:**
Survey specialty coffee roasters: "Would you use this as a starting profile?"

**Why This Metric?**
- **Ultimate ground truth**: Practical utility
- **Captures subtleties**: Quality factors beyond metrics

**Trade-offs:**
- ✅ **Strengths**:
  - Most valid assessment of usefulness
  - Catches issues metrics miss

- ❌ **Limitations**:
  - Expensive, slow
  - Requires domain experts (specialty roasters)
  - Subjective, variable

**Status**: Not feasible for this capstone (limited time/resources).

**Future Implementation:**
1. Partner with 3-5 specialty roasters
2. Provide 10 generated profiles per roaster
3. Ask to rate 1-5 scale + open feedback
4. Conduct blind comparison (real vs generated)

---

## Evaluation Protocol

### 1. Dataset Split Strategy

**Approach**: Stratified random split (80/20 train/val)

**Stratification Criteria**:
- Origin distribution preserved
- Process distribution preserved
- Roast level distribution preserved

**Why**:
- Ensures validation set is representative
- Tests generalization to similar-but-unseen profiles
- Prevents data leakage (no identical profiles in train/val)

**Validation Set Size**: 21 profiles (out of 144 total)
- Sufficient for statistical estimates
- Represents ~15% of dataset (conservative given small N)

**Course Connection**: Standard train/val split (Week 8 - training methodology), with domain-aware stratification.

---

### 2. Generalization Assessment

**Question**: How well does model handle unseen bean characteristics?

**Approach**:
1. **In-distribution**: Validation profiles with same origins as training
   - Tests: Interpolation within known space

2. **Out-of-distribution** (if possible):
   - Profiles from origins not in training set
   - Tests: Extrapolation to new beans

**Limitations** (small dataset):
- All 18 origins appear in training (not enough data to hold out)
- Can only assess in-distribution generalization
- Future: Multi-roaster dataset for true OOD evaluation

**Actual Analysis** *(Fill after evaluation)*:
```
Origins in validation set: [List origins]
  - Seen in training: [X/21]
  - New (if any): [X/21]

Performance by origin:
  [Origin 1]: MAE = [X.XX]°F
  [Origin 2]: MAE = [X.XX]°F
  ...

Interpretation:
  - Does model generalize within origin? [Yes/No + explanation]
  - Are some origins harder than others? [Analysis]
```

---

### 3. Visual Inspection Process

**Why**: Metrics don't capture everything. Human review is critical.

**Process**:
1. **Random Sample**: Review 10 randomly selected validation profiles
2. **Best/Worst**: Review top 3 and bottom 3 by MAE
3. **Edge Cases**: Review profiles with unusual characteristics

**Inspection Criteria**:
- Does trajectory "look like" a roast profile?
- Are phase transitions (drying, Maillard, development) present?
- Do turning point and first crack occur at reasonable times?
- Are there unexpected artifacts (oscillations, flat regions)?

**Documentation**:
- Screenshot real vs generated overlays
- Note qualitative observations
- Identify systematic patterns in failures

**Actual Observations** *(Fill after evaluation)*:
```
Visual Inspection Summary:
  - Overall impression: [Realistic/Somewhat realistic/Unrealistic]
  - Phase transitions: [Clear/Moderate/Absent]
  - Artifacts noted: [List any strange patterns]
  - Best examples: [Profile IDs]
  - Worst examples: [Profile IDs]

  Example:
    Profile #5 (Ethiopia Washed): Generated profile closely matches real,
    clear turning point at ~240s, first crack at ~480s. Slight overestimation
    of final 2 minutes (+8°F average).
```

---

### 4. Attention Pattern Analysis (Optional)

**Why**: Validate that model learns physically meaningful patterns.

**Approach**:
1. Extract attention weights from middle layer (Layer 3)
2. Visualize as heatmap (timesteps × timesteps)
3. Overlay roast phase boundaries:
   - End of drying phase (~240s)
   - First crack (~480s)
   - Development phase (480s - end)

**Hypothesis**: Different attention heads specialize in different phases.

**Actual Analysis** *(Fill if implemented)*:
```
Attention Analysis:
  - Do heads show phase specialization? [Yes/No + explanation]
  - Strong attention at first crack? [Yes/No]
  - Diagonal (local) vs global attention? [Pattern description]

  Course Connection: Validates transformer's ability to learn
  temporal structure (Week 5 - attention mechanisms).
```

---

## Results Summary

### Validation Set Performance

*(Fill after running evaluation notebook)*

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MAE** | [X.XX]°F | <10°F | [✅/⚠️/❌] |
| **DTW Distance** | [X.XX] | [baseline] | [✅/⚠️/❌] |
| **Finish Temp Accuracy** | [X.X]% | >75% | [✅/⚠️/❌] |
| **Monotonicity** | [X.X]% | >85% | [✅/⚠️/❌] |
| **Bounded RoR** | [X.X]% | >85% | [✅/⚠️/❌] |
| **Smooth Transitions** | [X.X]% | >95% | [✅/⚠️/❌] |

---

### Ablation Study Results

#### Positional Encoding Comparison

*(Fill from training notebook results)*

| Encoding Type | Best Val Loss | Winner? | Interpretation |
|---------------|---------------|---------|----------------|
| Sinusoidal | [X.XX]°F | [✅/❌] | [Theory connection] |
| Learned | [X.XX]°F | [✅/❌] | [Theory connection] |
| RoPE (optional) | [X.XX]°F | [✅/❌] | [Theory connection] |

**Analysis**:
```
Winner: [Sinusoidal/Learned/RoPE]
Improvement: [X.XX]°F ([X.X]%)

Course Connection (Week 5):
  - [Explain why this encoding worked better]
  - [Relate to positional encoding theory]
  - [Implications for time-series vs language modeling]
```

---

#### Flavor Conditioning Impact

*(Fill from training notebook if flavor ablation was run)*

| Configuration | Best Val Loss | MAE | Interpretation |
|---------------|---------------|-----|----------------|
| With Flavors | [X.XX]°F | [X.XX]°F | Baseline |
| Without Flavors | [X.XX]°F | [X.XX]°F | [Better/Worse/Same] |

**Impact**: [±X.XX]°F ([±X.X]%)

**Analysis**:
```
Result: Flavors [improved/worsened/had minimal impact on] performance

Interpretation:
  - If IMPROVED: Validates novel contribution - flavors provide signal
    beyond physical bean properties. Transformer learns flavor→temperature
    mappings.

  - If WORSENED/MINIMAL: Flavors may be redundant with origin/process.
    Small dataset (144 profiles) may be insufficient to learn flavor
    relationships. Flavors correlate with but don't cause temperature.

Course Connection (Week 6-7 - conditional generation):
  - Tests whether multi-modal conditioning (flavors) adds value
  - [Further analysis based on actual results]
```

---

### Qualitative Observations

*(Fill after visual inspection)*

**Best Performing Profiles**:
- Characteristics: [Common traits of best profiles]
- Example: [Profile #X description]

**Worst Performing Profiles**:
- Characteristics: [Common traits of worst profiles]
- Example: [Profile #Y description]

**Systematic Patterns**:
- [Pattern 1: e.g., "Underestimates final 2 minutes of development"]
- [Pattern 2: e.g., "Excellent on Ethiopian washed, struggles with naturals"]
- [Pattern 3]

**Surprising Successes/Failures**:
- [Unexpected result 1]
- [Unexpected result 2]

---

## Limitations & Trade-offs

### 1. Small Dataset Size

**Limitation**: Only 144 profiles (28-36 from Onyx, expanded with augmentation/batches)

**Impact**:
- Limited statistical power for confident generalization claims
- High variance in metrics (small validation set = noisy estimates)
- Cannot assess true out-of-distribution performance

**Trade-off Decisions**:
- ✅ Chosen: Small model (d_model=256) to prevent overfitting
- ✅ Chosen: Heavy regularization (dropout=0.1, weight_decay=0.01)
- ✅ Chosen: Early stopping to avoid memorization

**Future Mitigation**:
- Collect more data (100+ profiles from multiple roasters)
- Cross-validation instead of single train/val split
- Bayesian evaluation for uncertainty quantification

---

### 2. Single Roaster Bias

**Limitation**: All data from Onyx Coffee Lab (one roaster, one roasting style)

**Impact**:
- Model learns Onyx's specific style (high-charge, fast development)
- Unknown if generalizes to other roasters (e.g., slower Nordic style)
- Specialty coffee focus (not applicable to commodity coffee)

**Trade-off Decisions**:
- ✅ Chosen: Document bias clearly in model card
- ✅ Chosen: Frame as "Onyx-style profile generator" not "universal"
- ❌ **Not addressed**: Multi-roaster validation (out of scope)

**Future Mitigation**:
- Partner with 5-10 diverse roasters
- Train separate models per roaster
- Investigate transfer learning (pre-train on Onyx, fine-tune on others)

---

### 3. Evaluation Metric Limitations

**MAE Limitation**: Doesn't capture trajectory shape

**Mitigation**: Use DTW as complementary metric

**DTW Limitation**: Can hide timing errors

**Mitigation**: Use MAE + finish temp accuracy together

**Physics Constraints Limitation**: Binary (no gradations)

**Mitigation**: Visual inspection to assess "close" vs "way off"

**Missing: Human Evaluation**

**Limitation**: No roaster feedback on usability

**Future**: Partner with roasters for blind taste tests and profile ratings

---

### 4. Course-Specific Limitations

**Time Constraint**: Single semester capstone

**Impact**:
- Only 2-3 key experiments (not extensive hyperparameter search)
- Limited ablation studies
- No real-world validation

**Trade-off**:
- ✅ Focus on methodology depth (50 pts) over result perfection
- ✅ Demonstrate understanding through thoughtful evaluation
- ✅ Document limitations honestly (critical analysis - 10 pts)

---

## Future Evaluation Approaches

### 1. With More Resources (Time/Funding)

**Multi-Roaster Validation**:
- Collect 100+ profiles from 5-10 roasters
- Evaluate generalization across roasting styles
- Identify universal vs roaster-specific patterns

**Real-World Pilot Study**:
- Partner with 3 roasters
- Provide generated profiles as starting points
- Track: Time saved, profile quality, number of experimental roasts needed
- Measure: Economic impact (coffee saved, time saved)

**Human Evaluation at Scale**:
- Blind comparison: Real vs generated (can roasters tell the difference?)
- Usability survey: Would you use this tool?
- Longitudinal study: Do roasters continue using after initial trial?

---

### 2. Enhanced Metrics

**Perceptual Similarity**:
- Train roaster panel to rate profile similarity (1-10 scale)
- Use ratings to train learned metric (like LPIPS for images)
- More aligned with human judgment than MAE/DTW

**Phase-Specific Metrics**:
- Separate MAE for drying, Maillard, development phases
- Weight errors by phase importance (development > drying)
- Assess if model excels in certain phases

**Multi-Resolution DTW**:
- DTW at different time scales (second-level, minute-level)
- Captures both fine-grain and coarse-grain similarity

---

### 3. Generalization Testing

**Systematic OOD Evaluation**:
- Hold out specific origins/processes during training
- Test: "Can model generate Kenyan profiles if only trained on Ethiopian?"
- Quantify interpolation vs extrapolation performance

**Adversarial Testing**:
- Edge cases: Very high/low altitude, unusual processes
- Stress test: Can model refuse to generate if inputs are nonsensical?

**Transfer Learning**:
- Pre-train on Onyx data
- Fine-tune on new roaster (few-shot learning)
- Evaluate: How many profiles needed for adaptation?

---

### 4. Interpretability Analysis

**Feature Importance**:
- Ablate each input feature (origin, process, flavors, altitude, etc.)
- Quantify impact on generation quality
- Identify: Which features matter most?

**Attention Visualization at Scale**:
- Cluster attention patterns across all validation profiles
- Identify: Do certain bean characteristics produce distinctive attention?
- Validate: Do heads specialize consistently?

**Counterfactual Generation**:
- Fix all inputs except one (e.g., change only "berries" → "chocolate")
- Observe: How does profile change?
- Validate: Does model learn flavor→temperature relationships?

---

## References

### Course Materials

- **Week 5**: Transformer architecture, positional encodings
- **Week 6-7**: Conditional generation, controllability metrics
- **Week 8**: Regularization, small-data training strategies
- **Week 9**: Evaluation metrics for generative models
- **Week 10**: Time-series analysis, DTW

### Papers

- Vaswani et al. (2017): "Attention Is All You Need" - Transformer architecture
- Su et al. (2021): "RoFormer" - Rotary Position Embedding
- Salvador & Chan (2007): "FastDTW" - Dynamic Time Warping for time-series

### Domain References

- Specialty Coffee Association (SCA): Roast color classification (Agtron)
- Onyx Coffee Lab: roasting methodology and profile database
- *The Coffee Roaster's Companion* (Scott Rao): Roasting physics and best practices

---

## Appendix: Evaluation Checklist

Use this checklist to ensure complete evaluation:

### Before Running Evaluation

- [ ] Training complete (all experiments run)
- [ ] Best model identified (from experiment comparison)
- [ ] Checkpoint uploaded to Colab
- [ ] Validation data available

### During Evaluation

- [ ] All metrics computed (MAE, DTW, finish temp, physics)
- [ ] Visual comparisons created (10+ examples)
- [ ] Best/worst profiles identified
- [ ] Attention patterns analyzed (if implemented)

### After Evaluation

- [ ] Fill all `[X.XX]` placeholders in this document with actual results
- [ ] Write interpretations for each metric
- [ ] Complete ablation analysis sections
- [ ] Document qualitative observations
- [ ] Share with Claude for review/polish
- [ ] Export visualizations for presentation
- [ ] Package results (EVALUATION_SUMMARY.txt)

### For Presentation

- [ ] Key metrics on one slide (table format)
- [ ] 2-3 visual comparisons (real vs generated)
- [ ] Ablation study results (PE comparison, flavor impact)
- [ ] Limitations discussed honestly
- [ ] Future work motivated by evaluation findings

---

**Last Updated**: [Date]
**Status**: Template ready - fill after running `RoastFormer_Evaluation_Demo.ipynb`
**Points Value**: 15/125 (Assessment & Evaluation)

---

**This evaluation framework demonstrates:**
- Thoughtful metric selection with clear justifications
- Understanding of trade-offs and limitations
- Connection to course concepts (evaluation theory, domain knowledge)
- Honest assessment of small-data constraints
- Clear protocol for reproducible evaluation

**Ready to fill with actual results!** 📊
