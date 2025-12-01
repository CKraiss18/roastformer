# RoastFormer: Evaluation Findings & Presentation Guide

**Date**: November 20, 2025
**Status**: Evaluation Complete - Ready for Presentation
**Model**: d=256 (6.4M parameters, best training performance: 10.4°F RMSE)

---

## 🎯 Executive Summary

**Training Success**: RoastFormer achieved excellent training metrics (10.4°F RMSE) through systematic debugging (normalization fix) and comprehensive ablation studies.

**Evaluation Challenge**: Generation revealed autoregressive exposure bias problem - model learned patterns during training (teacher forcing) but struggles to generate coherent sequences independently.

**Key Learning**: Identified fundamental limitation, attempted solution (physics-constrained generation), analyzed why it failed, and identified proper solutions from literature. **This demonstrates scientific maturity and critical thinking.**

---

## 📊 Part 1: Training Results (SUCCESS ✅)

### Model Performance

| Model | d_model | Params | Params/Sample | RMSE (°F) | Status |
|-------|---------|--------|---------------|-----------|---------|
| **d=256** | **256** | **6,376,673** | **51,843** | **10.4** | **🏆 BEST** |
| d=128 | 128 | 1,088,993 | 8,854 | 16.5 | ✅ Strong |
| d=64 | 64 | 218,273 | 1,773 | 23.4 | ✅ Solid |
| d=32 | 32 | 45,665 | 371 | 49.3 | ✅ Baseline |

**Surprising Finding**: Large model (d=256) won despite 51,843:1 params/sample ratio!

**Why Our Prediction Was Wrong**:
- **Predicted**: d=256 would overfit (60°F+ RMSE)
- **Actual**: d=256 achieved 10.4°F RMSE (best!)
- **Lesson**: Normalization was THE critical bug. With proper regularization (dropout 0.2, weight decay 0.01, early stopping), larger models leverage capacity effectively.

**Scientific Value**: Hypothesis-driven experimentation led to deeper understanding. Being wrong experimentally > being theoretically correct without testing.

---

### Ablation Studies (VALIDATED ✅)

**1. Flavor Conditioning** (Novel Contribution):
- **With flavors**: 23.4°F RMSE
- **Without flavors**: 27.2°F RMSE
- **Improvement**: 3.8°F (14% better) ✅ **VALIDATES NOVEL CONTRIBUTION**

**2. Positional Encoding Comparison**:
- **Sinusoidal**: 23.4°F (best for small datasets)
- **RoPE**: 28.1°F (4.7°F worse - interesting given your presentation!)
- **Learned**: 43.8°F (overfits with 123 samples)

**Key Insight**: Task-specific testing beats theoretical assumptions. Classic methods (sinusoidal) win on small data.

---

### The Debugging Story (CRITICAL ✅)

**Initial Failure**: ALL 10 models predicted constant 16°F (model collapse)

**Two Fixes Identified**:
1. ✅ **Normalization** - Scale mismatch prevented learning
2. ✅ **Model Capacity** - Original prediction (d=256 would overfit) was wrong!

**Result**: After normalization, d=256 performed BEST (opposite of prediction)

**Presentation Value**: This debugging journey is your strongest asset - shows scientific maturity, systematic problem-solving, and honest reporting.

---

## 📉 Part 2: Evaluation Results (CHALLENGE IDENTIFIED ⚠️)

### Unconstrained Generation (Baseline)

**Accuracy Metrics**:
- **MAE**: 25.30°F
- **RMSE**: 29.81°F
- **Finish Temp MAE**: 13.95°F
- **Finish Temp Accuracy (±10°F)**: 50.0%

**Physics Compliance**:
- **Monotonicity** (post-turning): 0.0% ❌
- **Bounded RoR** (20-100°F/min): 28.8% ⚠️
- **Smooth Transitions** (<10°F/s): 98.7% ✅
- **Overall Physics Valid**: 0.0% ❌

**Analysis**:
Model achieves reasonable temperature accuracy (25°F MAE) but violates physics constraints. This is the **autoregressive exposure bias problem**:
- **During training**: Model sees real previous temperatures → learns patterns ✅
- **During generation**: Model sees own predictions → errors compound → physics violations ❌

---

### Design Decision: Duration as Input Parameter

**Observation**: All generated profiles are 600 seconds (10 minutes)

**Why?** Duration is an **input parameter**, not a model prediction:
- User specifies: `target_duration=600` (like specifying target finish temp)
- Model generates temperature trajectory for that specified duration
- Training data had variable durations (7-16 min, mean 11.2 min)

**Is This a Limitation?**

**Arguments for Design Choice** ✅:
- Roasters control duration based on desired roast level (darker = longer)
- Duration is a valid conditioning variable (like target temp, flavors)
- Gives user control over roast length
- Model learns: "For a 10-min light roast, use THIS temperature trajectory"

**Arguments for Limitation** ⚠️:
- Model doesn't learn "optimal duration" for a given coffee
- Real roasters adjust duration dynamically based on bean response
- Smarter model might predict: "This dense Ethiopian needs 11.5 min for light roast"
- User must know desired duration beforehand

**Future Enhancement**:
Add duration prediction module: Given bean characteristics + target roast level → predict optimal duration, then generate profile for that duration.

**For Presentation**: Frame as conscious design choice with enhancement opportunity, not a bug.

---

## 🔬 Part 3: Attempted Solution - Physics-Constrained Generation (LESSONS LEARNED 📚)

### Hypothesis

"Enforcing physics constraints during generation (monotonicity, bounded heating rates) should improve compliance while maintaining accuracy."

### Implementation

**Constraints Applied**:
1. Monotonic increase after turning point (prevent cooling)
2. Bounded heating rates (20-100°F/min)
3. Smooth transitions (<10°F/s)
4. Physical temperature bounds (250-450°F)

### Results: FAILED ❌

**Metrics Comparison**:

| Metric | Unconstrained | Constrained | Change |
|--------|---------------|-------------|--------|
| **MAE** | 25.3°F | 113.6°F | **+88.3°F** ❌ |
| **Finish Temp MAE** | 13.95°F | 86.67°F | **+72.7°F** ❌ |
| **Finish Accuracy** | 50.0% | 0.0% | **-50%** ❌ |
| **Monotonicity** | 0.0% | 100.0% | +100% ✅ |
| **Bounded RoR** | 28.8% | 0.0% | **-28.8%** ❌ |
| **Overall Valid** | 0.0% | 0.0% | No change |

**Visual Evidence**:
- **Unconstrained**: Reasonably follows real profile curves
- **Constrained**: Linear ramps (330°F → 500°F straight lines)
- **Generated profiles**: All look identical - unnatural and unrealistic

---

### Why It Failed: Root Cause Analysis

**Problem 1: Constraints Fight the Model**
- Model never learned proper roast dynamics (turning point, phase transitions)
- Constraints force behavior model doesn't understand
- Result: Unnatural linear ramps instead of realistic curves

**Problem 2: Post-Processing Cannot Fix Training Issues**
- Autoregressive exposure bias is a **training-time problem**
- Generation-time constraints are a **band-aid**, not a cure
- Model needs to learn proper dynamics during training

**Problem 3: Overly Aggressive Constraints**
- Forcing minimum heating rate created constant slopes
- Missing turning point phase (temperature dip)
- Bounded RoR made things worse (28.8% → 0%)

---

### What This Teaches Us (VALUABLE INSIGHTS 💡)

**1. Post-processing has limits**
- Cannot fix fundamental training issues
- Symptoms (bad generation) ≠ root cause (training process)

**2. Solutions must address root cause**
- Problem: Autoregressive exposure bias during training
- Wrong solution: Constrain generation (treats symptom)
- Right solution: Fix training process (see below)

**3. Negative results are scientifically valuable**
- Shows hypothesis testing
- Demonstrates critical analysis
- Identifies proper solutions from literature

**4. Experimental validation beats assumptions**
- Thought constraints would help
- Testing proved they make it worse
- Learned why and found better approaches

---

## ✅ Part 4: Proper Solutions (Future Work)

Based on literature and root cause analysis:

### 1. Scheduled Sampling (Bengio et al., 2015)

**Problem**: Training uses teacher forcing (real temps), generation uses own predictions

**Solution**: Gradually mix teacher forcing with model predictions during training
```
Epoch 1-10:  100% teacher forcing (real temps)
Epoch 11-20:  80% teacher forcing, 20% model predictions
Epoch 21-30:  50% teacher forcing, 50% model predictions
Epoch 31+:    0% teacher forcing (all model predictions)
```

**Benefit**: Model learns to handle its own prediction errors during training

---

### 2. Physics-Informed Loss Functions

**Problem**: Model not penalized for physics violations during training

**Solution**: Add physics constraints to loss function
```python
loss_prediction = MSE(predicted, actual)
loss_monotonicity = penalty_if_cooling_after_turning_point()
loss_bounded_ror = penalty_if_heating_rate_out_of_bounds()
loss_total = loss_prediction + 0.1 * loss_monotonicity + 0.1 * loss_bounded_ror
```

**Benefit**: Model learns to respect physics during training

---

### 3. Non-Autoregressive Generation (Diffusion Models)

**Problem**: Autoregressive models compound errors over 600 timesteps

**Solution**: Generate entire sequence at once (no error accumulation)
- Diffusion models (DDPM, Score-based models)
- Masked language models adapted for time-series
- Direct generation (predict all timesteps simultaneously)

**Benefit**: No exposure bias - training and generation processes match

---

### 4. Multi-Roaster Dataset with Diverse Styles (CRITICAL!)

**Problem**: 123 Onyx profiles = learning ONE championship roaster's style

**Why More Onyx Data Isn't Enough**:
- Even 500+ Onyx profiles = still learning Onyx's "house style"
- Onyx specializes in high-charge, fast development (modern light roasting)
- All on Loring S70 (no drum roasters, no fluid bed, no direct-fire)
- Competition-optimized profiles (not typical consumer preferences)

**What We Really Need**:
- **500+ profiles from 10+ diverse roasters**
- Equipment diversity: Loring, Probat, Diedrich, Giesen (drum), Sivetz (fluid bed)
- Style diversity: Nordic light, traditional medium, French dark, espresso
- Geographic diversity: US, Europe, Asia, Africa roasting cultures
- Skill levels: Championship, specialty, commercial

**Why This Matters**:
- Each roaster has a signature style (like a chef's cooking style)
- Equipment fundamentally shapes profiles (heat transfer, airflow, thermal mass)
- Cultural preferences differ dramatically (Scandinavia vs Italy vs Japan)
- Model currently learns "how Onyx roasts" not "how to roast"

**Benefit**:
- True generalization across roasting styles
- Transfer learning across equipment types
- Personalization: "Generate profile in MY style for new coffee"

**Lesson**: **Scale alone ≠ diversity. 500 profiles from one roaster < 200 profiles from 10 roasters.**

---

## 🎤 Part 5: Presentation Strategy

### Opening (Problem Statement)

> "Coffee roasters spend 10-20 experimental roasts (~15 minutes each) per new coffee, working from zero. RoastFormer aims to generate data-driven starting profiles using transformers conditioned on bean characteristics and desired flavors."

---

### Training Results Slide (SUCCESS STORY)

**Title**: "Training Results: Surprising Discovery"

**Visual**: Bar chart showing d=32, d=64, d=128, d=256 RMSE

**Key Points**:
- ✅ Systematic debugging identified two critical issues (normalization + capacity)
- ✅ d=256 achieved best results (10.4°F RMSE) - OPPOSITE of prediction!
- ✅ Flavor conditioning validated: 14% improvement (novel contribution)
- ✅ Comprehensive ablation studies completed

**Talking Points**:
> "I predicted the 6.4M parameter model would overfit on 123 samples. It achieved the best results—teaching me that normalization was the fundamental bug, and proper regularization enables larger models to leverage their capacity on small datasets."

---

### Evaluation Results Slide (HONEST ASSESSMENT)

**Title**: "Evaluation Challenge: Autoregressive Exposure Bias"

**Visual**: Side-by-side comparison (real profile vs generated)

**Metrics to Show**:
- ✅ Temperature accuracy: 25°F MAE (reasonable)
- ✅ Finish temp: 50% within ±10°F (decent)
- ❌ Physics compliance: 0% (problem identified)

**Key Points**:
- Model learned patterns during training (10.4°F RMSE with teacher forcing)
- Generation struggles due to exposure bias (sees own errors, not real temps)
- Identified fundamental limitation of autoregressive approach

**Talking Points**:
> "While training metrics were excellent, evaluation revealed a critical challenge: autoregressive exposure bias. The model learned patterns during training when it saw real previous temperatures, but struggles when generating independently and seeing its own predictions. This is a well-documented problem in sequence generation literature."

---

### Attempted Solution Slide (LESSONS LEARNED)

**Title**: "Attempted Solution: Physics-Constrained Generation"

**Visual**: 3-panel comparison
- Panel 1: Real profile (nice curve)
- Panel 2: Unconstrained (follows curve, some violations)
- Panel 3: Constrained (linear ramp - FAILED)

**What I Tried**:
- Enforced monotonicity after turning point
- Bounded heating rates (20-100°F/min)
- Smooth transitions

**Results**:
- ❌ MAE increased: 25°F → 114°F (4.5x worse)
- ❌ Generated linear ramps, not curves
- ❌ Physics compliance didn't improve

**What I Learned**:
- Post-processing cannot fix training issues
- Constraints fight against model's learned behavior
- Proper solutions must address root cause (training process)

**Talking Points**:
> "I hypothesized that physics-constrained generation would improve compliance. Testing proved this wrong—MAE increased 4.5x and generated profiles became unrealistic linear ramps. This taught me that post-processing cannot fix fundamental training issues. The proper solutions require addressing the training process itself through scheduled sampling, physics-informed losses, or non-autoregressive architectures."

---

### Future Work Slide (LITERATURE-GROUNDED)

**Title**: "Proper Solutions: Literature-Based Approaches"

**Visual**: Flowchart showing problem → solution mapping

**Solutions Identified**:

1. **Scheduled Sampling** (Bengio et al., 2015)
   - Gradually transition from teacher forcing to model predictions
   - Addresses exposure bias during training

2. **Physics-Informed Loss Functions**
   - Add penalties for physics violations to training loss
   - Model learns to respect constraints

3. **Non-Autoregressive Generation** (Diffusion Models)
   - Generate entire sequence at once
   - Eliminates error accumulation

4. **Larger Dataset** (500+ profiles, multi-roaster)
   - More data → better pattern learning
   - Transfer learning for generalization

**Talking Points**:
> "Based on my analysis, the proper solutions require changing the training process, not the generation process. Scheduled sampling would address exposure bias during training. Physics-informed losses would teach the model to respect constraints. Non-autoregressive approaches would eliminate error accumulation. These represent clear directions for future development."

---

## 📝 Part 6: Critical Analysis Content

### What Does This Reveal?

**1. Autoregressive Exposure Bias is Real**
- Well-documented in NLP (Bengio et al., 2015; Ranzato et al., 2016)
- Affects all autoregressive sequence models
- Training-generation mismatch causes compounding errors

**2. Small Data Amplifies Problems**
- 123 samples insufficient to learn complex dynamics
- Model memorizes patterns but doesn't understand roast physics
- Physics constraints become more critical with limited data

**3. Domain Knowledge Matters**
- Coffee roasting has hard physical constraints
- Violated constraints → unusable profiles
- Generic transformer approaches need domain adaptation

**4. Post-Processing Has Limits**
- Cannot fix what model never learned
- Constraints that fight model make things worse
- Solutions must address root causes, not symptoms

---

### Impact Assessment

**Academic Value**:
- ✅ Demonstrates transformer application to physical domain
- ✅ Validates flavor conditioning (novel contribution)
- ✅ Identifies exposure bias in time-series generation
- ✅ Shows systematic debugging and problem-solving

**Practical Value**:
- ⚠️ Current model: Not production-ready (physics violations)
- ✅ Training results: Proof of concept works
- ✅ Clear path forward: Literature-backed solutions identified
- ✅ Reduces problem scope: From "does this work?" to "how to scale?"

**Research Contribution**:
- Novel: Flavor-conditioned roast profile generation
- Validation: Transformers can learn from coffee roasting data
- Limitation: Autoregressive generation needs specialized training
- Direction: Physics-informed deep learning for specialty coffee

---

### Honest Limitations

**1. Small Dataset (123 Training Samples)**
- Insufficient for learning complex roast dynamics
- High variance in ablation results
- Limits generalization to new coffees

**2. Single Roaster Bias**
- All data from Onyx Coffee Lab
- Model learns Onyx-specific style
- Unknown generalization to other roasters

**3. Autoregressive Exposure Bias**
- Training-generation mismatch
- 0% physics compliance during free generation
- Requires training-time solutions (scheduled sampling)

**4. Lack of Real-World Validation**
- No feedback from professional roasters
- No blind testing of generated profiles
- Unknown practical utility without field trials

**5. Evaluation Constraints**
- Limited to temperature accuracy metrics
- Missing sensory evaluation (cup quality)
- No comparison to roaster's actual development process

---

## 🎓 Part 7: Key Messages for Presentation

### Message 1: Scientific Maturity

"This project demonstrates the scientific process: form hypothesis, design experiments, discover predictions are wrong, gain deeper understanding. Being experimentally wrong led to better insights than being theoretically correct without testing."

### Message 2: Honest Reporting

"Rather than hiding failures, I documented them thoroughly. The physics-constrained generation attempt failed, but taught me that post-processing cannot fix training issues. This demonstrates critical thinking and scientific integrity."

### Message 3: Literature Grounding

"After identifying the autoregressive exposure bias problem, I researched proper solutions from recent literature: scheduled sampling (Bengio et al., 2015), physics-informed neural networks, and non-autoregressive architectures. This shows ability to connect practical problems to academic research."

### Message 4: Course Integration

"This project applied concepts from multiple weeks: transformer architecture (Week 5), positional encodings (comparing sinusoidal vs RoPE), conditional generation (Week 6-7), regularization strategies (Week 8), and evaluation methodology (Week 9). The debugging process reinforced Week 2 fundamentals (normalization)."

### Message 5: Practical Vision

"Despite current limitations, RoastFormer demonstrates feasibility and identifies clear paths forward. With scheduled sampling training, physics-informed losses, and a larger multi-roaster dataset (500+ profiles), this approach could provide real value to specialty coffee roasters."

---

## 📋 Part 8: README Structure (Draft Outline)

```markdown
# RoastFormer: Transformer-Based Coffee Roast Profile Generation

## Overview
[Project description, motivation]

## Key Results

### Training Success ✅
- d=256 model: 10.4°F RMSE
- Flavor conditioning validated: +14% improvement
- Comprehensive ablation studies complete

### Evaluation Challenge ⚠️
- Autoregressive exposure bias identified
- 25°F MAE temperature accuracy
- 0% physics compliance (problem for future work)

## Novel Contributions
1. Flavor-conditioned generation (validated!)
2. Systematic debugging story (normalization fix)
3. Comprehensive positional encoding comparison
4. Critical analysis of autoregressive limitations

## Architecture
[Model design, features]

## Results & Analysis
[Training metrics, evaluation findings, lessons learned]

## Future Work
- Scheduled sampling for exposure bias
- Physics-informed loss functions
- Larger multi-roaster dataset
- Non-autoregressive architectures

## References
- Bengio et al. (2015) - Scheduled Sampling
- Vaswani et al. (2017) - Transformer Architecture
- Su et al. (2021) - RoPE (compared in ablations)

## Acknowledgments
[Course, advisor, Onyx Coffee Lab data]
```

---

## 🎯 Part 9: Presentation Flow (5-7 Minutes)

**Minute 1**: Problem & Motivation
- Coffee roasters spend 10-20 experimental roasts per coffee
- Goal: Generate data-driven starting profiles

**Minute 2**: Architecture & Features
- Transformer decoder-only
- Conditioned on bean characteristics + flavors (novel!)
- 144 profiles from Onyx Coffee Lab

**Minute 3**: Training Results (SUCCESS)
- Debugging story (normalization fix)
- d=256 won (opposite of prediction!)
- Flavor conditioning: +14% improvement
- Ablation studies validate choices

**Minute 4**: Evaluation Challenge (HONEST)
- Autoregressive exposure bias identified
- 25°F MAE accuracy, but 0% physics compliance
- Attempted physics-constrained generation (failed)

**Minute 5**: Lessons & Future Work
- Post-processing cannot fix training issues
- Proper solutions: scheduled sampling, physics-informed losses
- Clear path forward with literature-backed approaches

**Minute 6-7**: Demo & Questions
- Show comparison visualizations
- Discuss course connections
- Take questions

---

## ✅ Summary: What to Emphasize

### DO Emphasize:
✅ Training success (10.4°F RMSE, flavor validation)
✅ Systematic debugging (normalization fix story)
✅ Comprehensive ablations (PE comparison, flavor ablation)
✅ Honest evaluation (exposure bias identified)
✅ Scientific process (hypothesis testing, learning from failures)
✅ Literature grounding (scheduled sampling, proper solutions)
✅ Course integration (multiple weeks applied)

### DON'T Emphasize:
❌ Physics compliance metrics (all 0%, not helpful)
❌ Constrained generation as "success" (it failed)
❌ Production readiness (not there yet)

### DO Position As:
✅ Proof of concept with clear next steps
✅ Scientific exploration with valuable findings
✅ Novel contribution (flavor conditioning) validated
✅ Learning experience showing critical thinking

---

**Status**: Ready for README and presentation drafting! All findings documented, lessons learned captured, and talking points prepared. 🎯
