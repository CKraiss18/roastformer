# RoastFormer: Rubric & Course Concept Mapping

**Purpose**: Explicit mapping of evaluation findings to rubric requirements and course concepts
**Date**: November 20, 2024
**Target Score**: 110-120/125 points (88-96%)

---

## 📊 Point Tracking: Where We Stand

| Category | Points | Status | Evidence Document |
|----------|--------|--------|-------------------|
| **Methodology** | **50** | ✅ **SECURED** | `METHODOLOGY_COURSE_CONNECTIONS.md` (comprehensive) |
| Implementation & Demo | 20 | ✅ **SECURED** | Training notebook + Evaluation notebook |
| Assessment & Evaluation | 15 | ✅ **SECURED** | `EVALUATION_FINDINGS.md` (this adds detail) |
| Presentation | 10 | 📝 **READY** | `PRESENTATION_CHEAT_SHEET.md` + slides |
| Problem Statement | 10 | ✅ **DONE** | Proposal + CLAUDE.md |
| Critical Analysis | 10 | ✅ **READY** | `EVALUATION_FINDINGS.md` Part 6 |
| Documentation | 5 | ✅ **DONE** | README + comprehensive docs |
| Model & Data Cards | 5 | 📝 **TODO** | Quick writeup (30 min) |

**Current Estimated Score**: 110/125 (88%) - Excellent!
**With Model Card**: 115/125 (92%) - Outstanding!

---

## 🎯 Rubric Category 1: Problem Statement & Overview (10 pts) ✅

### Requirements:
- Problem clearly stated
- Proposed approach outlined
- Understandable presentation

### Evidence:
**Documents**:
- `Kraiss_Charlee_RoastFormer.pdf` (proposal)
- `CLAUDE.md` (comprehensive problem statement)
- `docs/EVALUATION_FINDINGS.md` Part 1 (executive summary)

**Presentation Opening** (from cheat sheet):
> "Coffee roasters spend 10-20 experimental roasts per new coffee—that's 2-3 hours just to find a starting profile. RoastFormer generates data-driven starting profiles using transformers conditioned on bean characteristics and desired flavors."

**Assessment**: **10/10 points** - Problem is crystal clear, approach is well-defined.

---

## 🎯 Rubric Category 2: Methodology (50 pts) ✅✅✅

**CRITICAL**: This is 40% of grade!

### Requirements:
- Course techniques/theories applied
- Clear connection to course content
- Theoretical foundations explained
- Methodology choices justified

### Course Concepts Applied (with Week Citations):

#### **Week 2: Neural Network Fundamentals**
**Applied**: Normalization discovery (critical bug fix)

**Methodology Connection**:
> "The debugging process revealed a fundamental Week 2 concept: input/output scale mismatch prevents learning entirely. Networks naturally output values near their initialization scale (0-10), but we asked for raw temperatures (150-450°F). This gradient explosion/vanishing prevented ALL learning until normalization was applied:
> ```python
> normalized_temp = (temp - 100) / (500 - 100)  # Map to [0,1]
> ```
> Result: 27x faster convergence, 11.5x better RMSE. This validates that preprocessing is more critical than architecture choices."

**Rubric Value**: Shows deep understanding of fundamentals, not just implementation.

---

#### **Week 4: Autoregressive Modeling**
**Applied**: Sequential temperature generation

**Methodology Connection**:
> "Following Week 4 autoregressive modeling theory (GPT-style), RoastFormer generates temperature sequences one step at a time:
> ```
> p(T₁, T₂, ..., Tₙ | conditions) = ∏ p(Tᵢ | T₁, ..., Tᵢ₋₁, conditions)
> ```
>
> **Advantage**: Matches physical causality (roasters see only past temps)
>
> **Challenge Identified**: Exposure bias (discussed in Week 4 lectures) - model trained with teacher forcing (sees real temps) but generates from own predictions (sees errors). This explains why training metrics (10.4°F RMSE with teacher forcing) don't translate to generation (25°F MAE with autoregressive generation)."

**Rubric Value**: Identifies course-covered limitation, shows understanding of theory-practice gap.

---

#### **Week 5: Transformer Architecture & Positional Encodings**
**Applied**: Decoder-only transformer, compared 3 PE methods

**Methodology Connection**:
> "Applied Week 5 transformer architecture (Vaswani et al., 2017, covered in lectures) with three positional encoding methods from course material:
>
> 1. **Sinusoidal** (original paper): 23.4°F RMSE ✅ Best
> 2. **RoPE** (Su et al., 2021, from my presentation): 28.1°F RMSE
> 3. **Learned** (alternative from course): 43.8°F RMSE ❌ Overfits
>
> **Key Finding**: Classic methods (sinusoidal) win on small datasets (123 samples), validating Week 5 discussion that zero-parameter encodings generalize better with limited data. Interesting that RoPE underperformed despite theoretical advantages for time-series—demonstrates importance of experimental validation over theoretical assumptions."

**Rubric Value**: Direct application of Week 5 concepts with experimental comparison, connects theory to results.

---

#### **Week 6-7: Conditional Generation**
**Applied**: Multi-modal feature conditioning (novel contribution!)

**Methodology Connection**:
> "Extended Week 6-7 conditional generation framework (P(x|c)) to multi-modal conditioning:
> - **Categorical** (origin, process, roast level) → Embeddings (Week 6 approach)
> - **Continuous** (altitude, finish temp) → Normalized + projected (standard practice)
> - **Set-valued** (flavors) → Multi-hot encoding → projected (**Novel contribution!**)
>
> **Ablation Study Result**: Flavor conditioning improved performance by 3.8°F (14%), validating that flavor features provide signal beyond physical bean characteristics. This validates the hypothesis that transformers can learn flavor→temperature mappings, demonstrating conditional generation on a novel feature type."

**Rubric Value**: Shows novel application of course concepts, validates with ablation study.

---

#### **Week 8: Small-Data Regime & Regularization**
**Applied**: Multiple regularization strategies, model size experiments

**Methodology Connection**:
> "Week 8 covered small-data training strategies. Applied four techniques:
> 1. **Dropout (0.2)**: Prevents co-adaptation (Srivastava et al., 2014, from course)
> 2. **Weight Decay (0.01)**: L2 regularization (standard from Week 8)
> 3. **Early Stopping (patience=15)**: Implicit regularization (Week 8 best practice)
> 4. **Model Size**: Tested d=32, 64, 128, 256
>
> **Surprising Result**: d=256 (6.4M params, 51,843:1 params/sample ratio) achieved BEST performance (10.4°F RMSE), opposite of prediction! This taught me that **normalization was THE critical bug**—with proper regularization, larger models leverage capacity effectively rather than overfitting.
>
> **Course Connection**: Validates Week 8 principle that multiple regularization strategies compound effectively, enabling larger models on small datasets."

**Rubric Value**: Systematic experimentation, surprising findings, deeper understanding from being wrong.

---

#### **Week 9: Evaluation Methodology**
**Applied**: Multi-metric evaluation framework

**Methodology Connection**:
> "Following Week 9 evaluation methodology, used multiple metrics beyond single-number performance:
>
> **Accuracy Metrics**:
> - MAE (25°F): Direct temperature accuracy
> - RMSE (30°F): Overall prediction quality
> - Finish Temp Accuracy (50%): Task success rate
>
> **Domain-Specific Metrics** (Week 9 concept: task-appropriate evaluation):
> - **Monotonicity**: Post-turning-point must increase (physics constraint)
> - **Bounded RoR**: 20-100°F/min heating rates (physical limits)
> - **Smooth Transitions**: <10°F/s changes (equipment constraints)
>
> **Result**: 0% physics compliance revealed fundamental limitation (autoregressive exposure bias), leading to attempted solution and valuable learning about post-processing vs training-time fixes."

**Rubric Value**: Shows understanding that evaluation is about methodology explanation, not just reporting numbers.

---

### Methodology Documentation:
- ✅ **Primary**: `METHODOLOGY_COURSE_CONNECTIONS.md` (comprehensive, 758 lines)
- ✅ **Applied**: `TWO_CRITICAL_FIXES.md` (debugging with theory)
- ✅ **Results**: `COMPREHENSIVE_RESULTS.md` (experiments with course connections)
- ✅ **Evaluation**: `EVALUATION_FINDINGS.md` (exposure bias analysis)

**Assessment**: **50/50 points** - Comprehensive course concept application with explicit week citations and theoretical justification.

---

## 🎯 Rubric Category 3: Implementation & Demo (20 pts) ✅

### Requirements:
- Code discussed and demonstrated
- Working examples shown
- OR pseudocode for theoretical aspects

### Evidence:
**Working Code**:
- `train_transformer.py` - Training pipeline
- `RoastFormer_Training_Suite.ipynb` - Comprehensive experiments (7 ablations)
- `RoastFormer_Evaluation_Demo_COMPLETE.ipynb` - Generation and evaluation
- `src/model/transformer_adapter.py` - Architecture implementation

**Demonstrated Results**:
- Training: d=256 (10.4°F RMSE), ablation studies complete
- Evaluation: 10 validation samples, metrics computed, visualizations created
- Demo: Interactive custom profile generation (in notebook)

**Presentation Demo Plan**:
- Live generation with custom inputs (backup: pre-generated examples)
- Visual comparisons (real vs generated profiles)
- Attention visualization (if time permits)

**Assessment**: **20/20 points** - Working implementation with comprehensive documentation and demo-ready notebooks.

---

## 🎯 Rubric Category 4: Assessment & Evaluation (15 pts) ✅

### Requirements:
- Approach assessed
- Evaluation explained
- Metrics justified
- Limitations discussed

### Evaluation Methodology Explanation:

#### **Why These Metrics?**

**Temperature Accuracy (MAE, RMSE)**:
> "We use MAE and RMSE to measure point-wise temperature accuracy because:
> - **Interpretable units** (°F) - roasters understand temperature
> - **Direct task relevance** - model must predict accurate temps
> - **Standard regression metrics** - allows comparison to baseline methods
>
> **Limitation**: Treats all timesteps equally, but first crack (380°F) is more critical than mid-roast temps."

**Finish Temperature Accuracy**:
> "We measure if generated profiles hit target finish temp (±10°F) because:
> - **Task success metric** - roasters specify target roast level
> - **Binary outcome** - clear success/failure
> - **Practical relevance** - finish temp determines roast level (light/medium/dark)
>
> **Limitation**: Could hit finish temp with terrible trajectory. Needs complementary metrics."

**Physics Compliance**:
> "We evaluate physics constraints because:
> - **Domain validity** - violations make profiles unusable
> - **Model understanding** - tests if model learned roast physics
> - **Failure mode detection** - identifies specific problems (monotonicity, heating rates)
>
> **Result**: 0% compliance identified autoregressive exposure bias as fundamental limitation."

#### **What Would We Do With More Resources?**

> "With more time/resources, evaluation would include:
> 1. **Multiple runs with different seeds** - quantify uncertainty (mean ± std dev)
> 2. **Cross-validation** - 5-fold CV instead of single train/val split
> 3. **Human evaluation** - roaster feedback surveys, blind comparisons
> 4. **Real-world validation** - test generated profiles on actual roasters
> 5. **Attention pattern analysis** - verify if model learned phase structure
> 6. **Larger test set** - 100+ profiles from multiple roasters for generalization check
>
> Current evaluation (10 samples, single split) is sufficient for proof-of-concept but limited for production claims."

### Limitations Honestly Discussed:
1. **Small dataset** (123 training samples) - insufficient for complex dynamics
2. **Single roaster bias** - all from Onyx, unknown generalization
3. **Autoregressive exposure bias** - 0% physics compliance in generation
4. **Lack of human validation** - no roaster feedback
5. **Limited evaluation set** - 10 samples, high variance

**Assessment**: **15/15 points** - Thorough evaluation methodology with justifications, limitations, and scaling discussion.

---

## 🎯 Rubric Category 5: Critical Analysis (10 pts) ✅

**Rubric asks for ONE OR MORE**:
1. What is the impact of this project?
2. What does it reveal or suggest?
3. What is the next step?

### 1. What Is The Impact?

**Academic Impact**:
> "RoastFormer demonstrates that transformers can learn from coffee roasting data and that flavor conditioning (novel contribution) provides meaningful signal (14% improvement). The debugging story (normalization fix, surprising d=256 result) contributes pedagogical value: experimental validation beats theoretical assumptions."

**Practical Impact** (Honest Assessment):
> "Current model is NOT production-ready (0% physics compliance makes profiles unusable). However, training success (10.4°F RMSE) proves concept feasibility. With proper training-time solutions (scheduled sampling, physics-informed losses) and larger dataset (500+ profiles), this approach could reduce roaster experimentation from 10-20 roasts to 2-3 validation roasts—saving 2-3 hours per new coffee."

**Research Impact**:
> "Identifies autoregressive exposure bias as fundamental challenge for physics-constrained generation, pointing to scheduled sampling (Bengio et al., 2015) and physics-informed neural networks as necessary solutions. Contributes to understanding of transformers for time-series generation in physically-constrained domains."

### 2. What Does It Reveal or Suggest?

**About Transformers for Time-Series**:
> "RoastFormer reveals that:
> 1. **Normalization is critical** - 27x faster convergence after fix, more important than architecture
> 2. **Small data ≠ small models** - With proper regularization, d=256 (51,843:1 ratio) won vs smaller models
> 3. **Classic methods win on small data** - Sinusoidal PE beat modern RoPE on 123 samples
> 4. **Multi-modal conditioning works** - Flavors improved performance 14%
> 5. **Training-generation gap is real** - 10.4°F RMSE (training) vs 25°F MAE (generation) with 0% physics compliance"

**About Autoregressive Exposure Bias**:
> "Evaluation revealed textbook example of exposure bias (Week 4 concept):
> - **Training**: Teacher forcing (model sees real temps) → learns patterns ✅
> - **Generation**: Own predictions (model sees errors) → errors compound ❌
>
> Attempted solution (physics-constrained decoding) FAILED (MAE 4.5x worse), teaching that post-processing cannot fix training issues. This demonstrates that solutions must address root causes (training process) not symptoms (generation output). Proper solutions require scheduled sampling or physics-informed losses."

**About Domain Adaptation**:
> "Coffee roasting has hard physical constraints (monotonicity, bounded heating rates) that generic transformers don't respect. Physics-informed approaches (inductive biases, constrained losses) are necessary for physically-constrained domains. This suggests that transformers for science/engineering applications need domain-specific adaptations beyond standard NLP/CV techniques."

### 3. What Is The Next Step?

**Immediate (Proof-of-Concept → Production)**:
1. **Scheduled Sampling Training** (Bengio et al., 2015)
   - Gradually transition from teacher forcing to model predictions
   - Addresses exposure bias at source
   - Standard technique from literature

2. **Physics-Informed Loss Functions**
   - Add penalties for monotonicity violations, bounded RoR violations
   - Model learns constraints during training
   - Inspired by physics-informed neural networks (PINNs)

3. **Larger Multi-Roaster Dataset** (500+ profiles)
   - More data → better pattern learning
   - Multiple roasters → generalization beyond Onyx style
   - Transfer learning → pre-train on large dataset, fine-tune on specific roaster

**Long-Term (Research Directions)**:
4. **Non-Autoregressive Architectures** (Diffusion Models)
   - Generate entire sequence at once
   - No error accumulation
   - State-of-the-art for sequence generation

5. **Interactive Generation**
   - Real-time roaster feedback during generation
   - Reinforcement learning from roaster preferences
   - Human-in-the-loop optimization

6. **Inverse Design**
   - Desired flavor profile → optimal roast trajectory
   - Optimization in latent space
   - Multi-objective optimization (flavor + yield + consistency)

**Assessment**: **10/10 points** - Comprehensive analysis covering impact, insights, and clear next steps grounded in literature.

---

## 🎯 Rubric Category 6: Model & Data Cards (5 pts) 📝

### Requirements:
- Model version/architecture shown
- Intended uses outlined
- Licenses outlined
- Ethical considerations addressed
- Bias considerations addressed

### Quick Template (30-minute task):

**Model Card** (`docs/MODEL_CARD.md`):
```markdown
# RoastFormer Model Card

## Model Details
- **Name**: RoastFormer v1.0
- **Architecture**: Transformer decoder-only (6 layers, d=256, 8 heads)
- **Parameters**: 6,376,673
- **Training Data**: 123 roast profiles from Onyx Coffee Lab
- **Performance**: 10.4°F RMSE (training), 25°F MAE (generation)
- **Date**: November 2024
- **License**: MIT

## Intended Use
- **Primary**: Educational/research proof-of-concept
- **NOT intended for**: Production roasting (0% physics compliance)
- **Potential future use**: Starting point generation with human validation

## Limitations
- Small dataset (123 samples, single roaster)
- 0% physics compliance in generation
- Unknown generalization to other roasters
- No human validation

## Ethical Considerations
- **Bias**: Trained only on Onyx style (championship specialty roaster)
- **Not representative**: High-end specialty coffee, not commodity
- **Safety**: Generated profiles should NOT be used without expert review

## Training Data
- **Source**: Onyx Coffee Lab (public roast profiles)
- **Attribution**: Onyx Coffee Lab, championship roasters
- **License**: Public data, used for educational purposes
```

**Data Card** (`docs/DATA_CARD.md`):
```markdown
# RoastFormer Dataset Card

## Dataset
- **Name**: Onyx Roast Profiles
- **Size**: 144 profiles (123 train, 21 val)
- **Source**: Onyx Coffee Lab public profiles
- **Date Range**: October-November 2024
- **Format**: JSON time-series + metadata CSV

## Features
- **Temperature sequences**: 1-second resolution, 400-600 points
- **Bean characteristics**: Origin, process, variety, altitude
- **Roast parameters**: Target finish temp, roast level
- **Flavor notes**: Multi-hot encoding, 98 unique flavors

## Bias Considerations
- **Single roaster**: All from Onyx (championship style)
- **Specialty focus**: High-end coffees, not representative
- **Geographic bias**: Primarily Ethiopia, Colombia, Central America
- **Process bias**: Mostly washed process (clean, bright profiles)

## Limitations
- Small sample size (not suitable for production models)
- No ground truth flavor measurements (cupping scores)
- Self-reported metadata (potential inaccuracies)
```

**Assessment**: **5/5 points** - Straightforward documentation, 30-minute task.

---

## 🎯 Rubric Category 7: Documentation & Resource Links (5 pts) ✅

### Requirements:
- Repository exists with README
- Setup instructions clear
- Resource links provided
- Relevant papers cited

### Evidence:
**Repository**: https://github.com/CKraiss18/roastformer

**Documentation Structure**:
- `README.md` - Project overview, setup, usage
- `CLAUDE.md` - Development guide (comprehensive)
- `docs/` - 25+ documentation files organized by category

**Key References Cited**:
- Vaswani et al. (2017) - "Attention Is All You Need"
- Su et al. (2021) - "RoFormer: Enhanced Transformer with RoPE"
- Bengio et al. (2015) - "Scheduled Sampling for Sequence Prediction"
- Srivastava et al. (2014) - "Dropout"
- Loshchilov & Hutter (2019) - "AdamW"

**Resource Links**:
- Onyx Coffee Lab data source
- PyTorch transformer tutorials
- Course materials references

**Assessment**: **5/5 points** - Comprehensive documentation with proper citations.

---

## 🎯 Rubric Category 8: Presentation (10 pts) 📝

### Requirements:
- Organization & Clarity (4 pts)
- Visual Aids & Demonstrations (3 pts)
- Delivery & Engagement (2 pts)
- Preparation & Professionalism (1 pt)

### Preparation:

**Organization (4 pts)**:
- Clear 5-minute flow (see `PRESENTATION_CHEAT_SHEET.md`)
- Problem → Architecture → Training Success → Evaluation Challenge → Lessons & Future Work
- Logical progression with course connections throughout

**Visual Aids (3 pts)**:
- Architecture diagram (transformer decoder + conditioning)
- Training results bar chart (d=32, 64, 128, 256)
- Real vs generated comparison plots
- Constrained generation failure (linear ramps - lessons learned)
- Future work flowchart (literature-grounded solutions)

**Delivery & Engagement (2 pts)**:
- Prepared talking points (see cheat sheet)
- Demo backup plan (pre-generated examples)
- Anticipated questions with answers ready

**Professionalism (1 pt)**:
- Clean slides with consistent formatting
- Proper citations
- Time management (5-7 minute slot)

**Assessment**: **10/10 points** - Well-prepared with comprehensive guide and backup materials.

---

## 📊 Final Score Projection

| Category | Points | Confidence |
|----------|--------|------------|
| Problem Statement | 10 | ✅ 100% |
| **Methodology** | **50** | ✅ **100%** |
| Implementation & Demo | 20 | ✅ 100% |
| Assessment & Evaluation | 15 | ✅ 100% |
| Model & Data Cards | 5 | 📝 95% (30-min task) |
| Critical Analysis | 10 | ✅ 100% |
| Documentation | 5 | ✅ 100% |
| Presentation | 10 | 📝 95% (practice needed) |

**Projected Score**: **115/125 (92%)** - Outstanding!

---

## 🎓 Key Strengths for Grading

### What Makes This Strong:

1. **Comprehensive Methodology** (50 pts):
   - Explicit course concept citations (Week 2, 4, 5, 6-7, 8, 9)
   - Theoretical justification for every choice
   - Surprising experimental findings with deeper insights

2. **Honest Scientific Reporting** (15+10 pts):
   - Documented failures (constrained generation)
   - Explained why failures occurred
   - Identified proper solutions from literature
   - Shows maturity over perfection

3. **Novel Contribution** (Methodology bonus):
   - Flavor conditioning validated (14% improvement)
   - First application of transformers to flavor-guided roast generation
   - Ablation study proves value

4. **Systematic Debugging** (Methodology bonus):
   - Normalization discovery (27x improvement)
   - Surprising d=256 result (opposite of prediction)
   - Demonstrates scientific process over correctness

5. **Literature Integration** (Multiple categories):
   - Bengio et al. (2015) - Scheduled sampling
   - Su et al. (2021) - RoPE (from your presentation!)
   - Vaswani et al. (2017) - Transformer architecture
   - Shows ability to connect practice to research

---

## 💡 Presentation Strategy: Hit These Points

### Opening (Hook):
"Coffee roasters spend 10-20 experimental roasts per new coffee. RoastFormer aims to provide data-driven starting profiles."

### Middle (Course Connections):
- "Applied Week 5 transformer architecture..."
- "Tested three positional encoding methods from Week 5..."
- "Identified Week 4 autoregressive exposure bias..."
- "Used Week 8 regularization strategies..."

### Closing (Scientific Maturity):
"Being experimentally wrong taught me more than being theoretically correct. The physics-constrained generation attempt failed, but revealed that post-processing cannot fix training issues—solutions must address root causes through scheduled sampling or physics-informed losses from literature."

---

## ✅ Final Checklist Before Presentation

**Documentation**:
- [x] Methodology document (comprehensive)
- [x] Evaluation findings (complete)
- [x] Presentation cheat sheet (ready)
- [x] Rubric mapping (this document)
- [ ] Model card (30-min task)
- [ ] Data card (30-min task)

**Materials**:
- [ ] Slide deck (5-7 slides)
- [ ] Visual aids (charts, diagrams)
- [ ] Demo notebook (tested, backup ready)
- [ ] Practice delivery (2-3 times)

**Key Messages**:
- [ ] Course connections explicit
- [ ] Novel contribution highlighted
- [ ] Honest limitations discussed
- [ ] Future work literature-grounded
- [ ] Scientific process demonstrated

---

**Bottom Line**: You have comprehensive documentation mapping to ALL rubric requirements with explicit course connections. The evaluation "challenges" (exposure bias, constrained generation failure) actually STRENGTHEN your presentation by showing scientific maturity, critical thinking, and literature awareness. This is A-level work. 🎯
