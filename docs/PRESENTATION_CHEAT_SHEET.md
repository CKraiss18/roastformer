# RoastFormer: Presentation Cheat Sheet

**5-Minute Presentation Guide** | Nov 20, 2025

---

## 🎤 Opening (30 seconds)

**Hook**: "Coffee roasters spend 10-20 experimental roasts per new coffee—that's 2-3 hours just to find a starting profile."

**Goal**: "RoastFormer generates data-driven starting profiles using transformers conditioned on bean characteristics and desired flavors."

**Data**: "144 profiles from Onyx Coffee Lab, championship-winning specialty roaster."

---

## 🏗️ Slide 1: Architecture (30 seconds)

**Model**: Transformer decoder-only, 6.4M parameters

**Novel Contribution**: Flavor-conditioned generation
- Origin, process, variety, altitude (standard)
- **+ Flavor notes** (berries, chocolate, floral, etc.) ← Novel!

**Quick mention**: "Evaluated 3 positional encoding methods including RoPE from the paper I presented."

---

## ✅ Slide 2: Training Success (90 seconds)

### Part A: The Debugging Story (45s)

**Initial Problem**: ALL 10 models failed (predicted constant 16°F)

**Systematic Debugging**:
1. Tried smaller models → still failed
2. Tried lower learning rates → still failed
3. Analyzed training logs → found root cause

**Root Cause**: Missing temperature normalization (scale mismatch)
- Targets: 150-450°F
- Network outputs: 0-10
- Result: Gradient explosion/vanishing

**Fix**: Normalize to [0,1] range → 27x faster convergence

### Part B: Surprising Result (45s)

**Hypothesis**: Large model (6.4M params, 51,843:1 ratio) would overfit

**Experiments**: Tested d=32, d=64, d=128, d=256

**Result**: d=256 WON! (10.4°F RMSE - best performance)

**Why I Was Wrong**:
> "Normalization was THE critical bug. With proper regularization (dropout, weight decay, early stopping), larger models leverage capacity to learn complex roast dynamics. Being experimentally wrong taught me more than being theoretically correct."

**Ablation Validation**:
- ✅ Flavor conditioning: +14% improvement (VALIDATES novel contribution!)
- ✅ Sinusoidal PE best (23.4°F vs RoPE 28.1°F) - interesting given my presentation!

---

## ⚠️ Slide 3: Evaluation Challenge (90 seconds)

### Part A: The Problem (30s)

**Metrics**:
- Temperature accuracy: 25°F MAE (reasonable)
- Finish temp: 50% within ±10°F (decent)
- **Physics compliance: 0%** ❌ (problem!)

**Root Cause**: Autoregressive exposure bias
- Training: Model sees real previous temps → learns patterns ✅
- Generation: Model sees own predictions → errors compound ❌

### Part B: Attempted Solution (60s)

**What I Tried**: Physics-constrained generation
- Enforce monotonicity (no cooling after turning point)
- Bound heating rates (20-100°F/min)
- Smooth transitions

**Results**: FAILED
- MAE: 25°F → 114°F (4.5x worse!)
- Generated linear ramps, not curves
- Physics compliance didn't improve

**What I Learned**:
> "Post-processing constraints cannot fix training issues. The constraints fought against the model's learned behavior, creating unnatural linear ramps. This taught me that solutions must address the root cause—the training process—not the symptoms."

**Visual**: Show comparison plot (real curve vs constrained linear ramp)

---

## 🚀 Slide 4: Future Work (60 seconds)

**Proper Solutions** (literature-backed):

1. **Scheduled Sampling** (Bengio et al., 2015)
   - Gradually transition from teacher forcing to model predictions during training
   - Addresses exposure bias at the source

2. **Physics-Informed Loss Functions**
   - Add penalties for physics violations to training loss
   - Model learns to respect constraints

3. **Multi-Roaster Dataset with Diverse Styles** (CRITICAL!)
   - **Not just more Onyx data** - need 10+ diverse roasters
   - Equipment diversity: Loring, Probat, Diedrich (drum), Sivetz (fluid bed)
   - Style diversity: Nordic light, traditional medium, French dark
   - Geographic diversity: US, Europe, Asia, Africa roasting cultures
   - **Key insight**: Scale alone ≠ diversity. 500 from one roaster < 200 from 10 roasters
   - Model currently learns "Onyx's style" not "how to roast"

4. **Duration Prediction Module**
   - Current: User specifies duration (design choice, like target temp)
   - Future: Model predicts optimal duration for coffee
   - "This dense Ethiopian at 2100m needs 11.5 min for light roast"

5. **Non-Autoregressive Architectures** (Diffusion Models)
   - Generate entire sequence at once
   - No error accumulation

**Closing**:
> "Despite current limitations, RoastFormer demonstrates feasibility of transformer-based profile generation, validates flavor conditioning as a meaningful feature, and identifies clear paths forward with literature-backed solutions. This represents a proof-of-concept with practical potential for specialty coffee."

---

## 📊 Backup Slides (If Time Allows)

### Course Connections

**Week 2**: Neural network fundamentals (normalization critical!)
**Week 5**: Transformer architecture, positional encodings (compared 3 methods)
**Week 6-7**: Conditional generation (flavor features)
**Week 8**: Small-data strategies, regularization (validated with d=256 success)
**Week 9**: Evaluation methodology (honest reporting of limitations)

### Statistics

- **Dataset**: 144 profiles (123 train, 21 val)
- **Training time**: <20 minutes per experiment on GPU
- **Total experiments**: 7 comprehensive ablations
- **Best model**: d=256 (6,376,673 parameters)
- **Novel contribution validated**: Flavors improve performance 14%

---

## 🎯 Key Messages to Hit

1. **Scientific Process**: Hypothesis → Experiment → Surprise → Understanding
2. **Honest Reporting**: Documented failures, learned from them
3. **Literature Grounding**: Connected problems to research solutions
4. **Course Integration**: Applied concepts from multiple weeks
5. **Practical Vision**: Clear path forward despite current limitations

---

## ❌ Common Pitfalls to Avoid

**DON'T**:
- Claim production-ready (it's not)
- Overemphasize physics compliance metrics (all 0%)
- Pretend constrained generation worked (it failed)
- Apologize excessively for limitations

**DO**:
- Emphasize training success (10.4°F RMSE, flavor validation)
- Frame failures as learning experiences
- Show understanding of root causes
- Position as proof-of-concept with clear next steps

---

## 🔥 If Asked Difficult Questions

### Q: "Why did constrained generation fail so badly?"

**A**: "Great question! The constraints tried to force physical behavior the model never learned during training. Because the model was trained with teacher forcing—seeing real previous temperatures—it never learned to handle its own prediction errors. The constraints fought against the model's learned behavior, making predictions worse. This taught me that the solution must fix the training process through scheduled sampling or physics-informed losses, not the generation process."

### Q: "Is this useful to roasters in its current state?"

**A**: "In its current form, no—the physics violations make generated profiles unreliable. However, the training results (10.4°F RMSE, flavor conditioning validation) prove the concept works. With proper training-time solutions like scheduled sampling and a larger dataset, this approach could provide real value. The current work identifies both feasibility and the specific obstacles to overcome."

### Q: "Why only 144 profiles?"

**A**: "Specialty coffee profile data is difficult to obtain—roasters guard their process data closely. Onyx Coffee Lab makes some profiles publicly available, giving me 144 samples. This limitation actually became a learning opportunity, demonstrating how small datasets amplify issues like exposure bias and making proper regularization critical (as shown by d=256's success)."

### Q: "How does this compare to existing methods?"

**A**: "Most roasters work from experience or simple curve templates. There's limited academic work on ML-driven profile generation. The novel contribution here is flavor conditioning—using desired sensory outcomes to guide generation. The validation (14% improvement) shows this approach has merit, even though the generation challenge revealed implementation hurdles."

---

## ⏱️ Timing Breakdown

- **Opening**: 30s
- **Architecture**: 30s
- **Training Success**: 90s
- **Evaluation Challenge**: 90s
- **Future Work**: 60s
- **Questions**: 30-60s

**Total**: 5-6 minutes (perfect for 5-7 minute slot)

---

## 🎨 Visual Checklist

Slides to have ready:
- [ ] Architecture diagram
- [ ] Training results bar chart (d=32, d=64, d=128, d=256 RMSE)
- [ ] Ablation results (flavors, PE comparison)
- [ ] Real vs generated comparison (show the problem)
- [ ] Constrained generation failure (linear ramps)
- [ ] Future work flowchart

---

## 💪 Confidence Boosters

**Remember**:
- Your debugging story is STRONG (normalization fix)
- The d=256 surprise result is INTERESTING (opposite of prediction)
- Flavor validation is YOUR NOVEL CONTRIBUTION (14% improvement!)
- Honest reporting of failures shows MATURITY
- Literature-backed solutions show DEPTH

**You have**:
- Comprehensive ablation studies (7 experiments)
- Systematic debugging methodology
- Validated novel contribution (flavors)
- Understanding of limitations and proper solutions
- Course concept integration across multiple weeks

**This is solid work!** Present with confidence. 🎯
