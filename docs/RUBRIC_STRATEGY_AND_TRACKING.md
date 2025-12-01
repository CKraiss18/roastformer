# Capstone Rubric Strategy & Tracking

**Created**: Nov 11, 2025
**Purpose**: Keep development focused on maximum point value
**Total Points**: 125

---

## 🎯 Critical Insight

**METHODOLOGY = 40% OF GRADE (50/125 points)**
**Must demonstrate course connections, NOT just implementation quality**

---

## 📊 Point Distribution & Current Status

| Category | Points | % | Priority | Status | Notes |
|----------|--------|---|----------|--------|-------|
| **Methodology** | **50** | **40%** | 🔥 CRITICAL | ⚠️ IN PROGRESS | Course connections document |
| Implementation & Demo | 20 | 16% | HIGH | ⚠️ IN PROGRESS | Basic Colab training |
| Assessment & Evaluation | 15 | 12% | HIGH | 📝 TODO | Evaluation framework writeup |
| Presentation | 10 | 8% | MEDIUM | 📝 TODO | Slides, visuals, demo |
| Problem Statement | 10 | 8% | LOW | ✅ DONE | Already strong |
| Critical Analysis | 10 | 8% | MEDIUM | 📝 TODO | Impact, insights, next steps |
| Documentation | 5 | 4% | LOW | ✅ MOSTLY DONE | Polish README, add citations |
| Model & Data Cards | 5 | 4% | LOW | 📝 TODO | Quick writeup at end |

**Current Estimated Score**: 20/125 (16%) - Problem statement only
**Target Score**: 110-120/125 (88-96%)

---

## 🚨 Common Pitfalls to AVOID

### DON'T Over-Invest On:
- ❌ Perfect training results (only matters for demo)
- ❌ Extensive ablation studies (not in rubric!)
- ❌ Multiple model configurations (one working model = full credit)
- ❌ Optimal hyperparameters (not separately graded)
- ❌ Production-ready code (pseudocode is acceptable!)

### DO Invest On:
- ✅ **Methodology connections to course content** (50 pts!)
- ✅ **Clear explanation of theoretical foundations** (methodology)
- ✅ **Visual aids and presentation materials** (13 pts combined)
- ✅ **Evaluation framework explanation** (15 pts)
- ✅ **Critical analysis of impact and insights** (10 pts)

---

## 📅 Revised 3-Week Timeline

### Week 1: Nov 11-15 (METHODOLOGY + BASIC TRAINING)
**Goal**: Secure 85/125 points (68% of grade)

#### Monday-Tuesday (Nov 11-12): Methodology (50 pts)
- [ ] Create `METHODOLOGY_COURSE_CONNECTIONS.md`
- [ ] Map transformer architecture → course lectures
- [ ] Explain attention mechanisms → roast profile theory
- [ ] Compare positional encodings (3 types from course)
- [ ] Conditioning approach → conditional generation theory
- [ ] Cite course materials, textbook, lectures
- [ ] Explain WHY each choice, not just WHAT

#### Wednesday-Thursday (Nov 13-14): Basic Training (20 pts)
- [ ] Simple Colab training pipeline
- [ ] Generate 2-3 example profiles (for demo)
- [ ] Basic metrics (MAE, finish temp accuracy)
- [ ] Document what works, what doesn't
- [ ] NO extensive tuning needed!

#### Friday (Nov 15): Evaluation Framework (15 pts)
- [ ] Write evaluation methodology explanation
- [ ] Justify metric choices (MAE, DTW, physics)
- [ ] Discuss trade-offs and limitations
- [ ] Explain how to assess with more resources

**Week 1 Checkpoint**: 85/125 points covered ✓

---

### Week 2: Nov 18-22 (ANALYSIS + PRESENTATION)
**Goal**: Cover remaining high-value items

#### Monday-Tuesday (Nov 18-19): Critical Analysis (10 pts)
- [ ] Impact: How does this help roasters?
- [ ] Novel contribution: Flavor-conditioned generation
- [ ] Insights: What does it reveal about transformers for time-series?
- [ ] Limitations: Small dataset, single roaster
- [ ] Next steps: More data, multi-roaster, real-world validation

#### Wednesday-Friday (Nov 20-22): Presentation Materials (10 pts)
- [ ] Slide deck structure (Organization & Clarity: 4 pts)
- [ ] Visual aids (3 pts):
  - Architecture diagram
  - Real vs generated profile plots
  - Attention visualization (if feasible)
  - Feature encoding illustration
- [ ] Demo plan with backup slides
- [ ] Practice delivery (Engagement: 2 pts)
- [ ] Professional polish (1 pt)

**Week 2 Checkpoint**: 105/125 points covered ✓

---

### Week 3: Nov 25-29 (POLISH + PRACTICE)
**Goal**: Final touches and preparation

#### Monday (Nov 25): Documentation Polish (5 pts)
- [ ] Model Card:
  - Architecture description
  - Intended uses
  - Licenses (MIT)
  - Ethical considerations
  - Bias considerations (single roaster, specialty focus)
- [ ] README polish:
  - Setup instructions
  - Usage guide
  - Clear examples
- [ ] Resource Links:
  - Attention Is All You Need paper
  - RoPE paper
  - Relevant transformer tutorials
  - Onyx Coffee Lab citation

#### Tuesday-Wednesday (Nov 26-27): Final Polish
- [ ] Review methodology document
- [ ] Presentation rehearsal
- [ ] Generate fresh example profiles
- [ ] Prepare backup materials

#### Thursday (Nov 28): Thanksgiving Break

#### Friday (Nov 29): Final Prep
- [ ] Last presentation practice
- [ ] Test demo backup plan
- [ ] Materials ready to submit

**Week 3 Checkpoint**: All 125 points addressed ✓

---

## 🎯 TODAY'S HYBRID APPROACH (Nov 11)

### Phase 1: Methodology Framework (1 hour) - 50 pts
- [ ] Create `METHODOLOGY_COURSE_CONNECTIONS.md` structure
- [ ] Outline key course concepts to connect
- [ ] Draft transformer theory sections

### Phase 2: Basic Colab Training (2-3 hours) - 20 pts
- [ ] Set up Colab environment
- [ ] Load preprocessed data
- [ ] Simple training loop
- [ ] Generate 1-2 example profiles
- [ ] Document basic results

### Phase 3: Evaluation Framework Start (1 hour) - 15 pts
- [ ] Draft evaluation methodology writeup
- [ ] Explain metric choices
- [ ] Initial limitations discussion

**Today's Target**: Framework for 85/125 points (68%)

---

## 📋 Rubric Requirements Checklist

### 1. Problem Statement & Overview (10 pts)
- [x] Problem clearly stated
- [x] Proposed approach outlined
- [x] Understandable presentation
**Status**: ✅ COMPLETE (via proposal and CLAUDE.md)

### 2. Methodology (50 pts) ⚠️ CRITICAL
- [ ] Course techniques/theories applied
- [ ] Clear connection to course content
- [ ] Theoretical foundations explained
- [ ] Methodology choices justified
**Status**: ⚠️ IN PROGRESS (highest priority)

### 3. Implementation & Demo (20 pts)
- [ ] Code discussed
- [ ] Code demonstrated
- [ ] Working examples shown
- [ ] OR pseudocode for theoretical aspects
**Status**: ⚠️ IN PROGRESS (Colab training)

### 4. Assessment & Evaluation (15 pts)
- [ ] Approach assessed
- [ ] Evaluation explained
- [ ] Metrics justified
- [ ] Limitations discussed
**Status**: 📝 TODO (this week)

### 5. Model & Data Cards (5 pts)
- [ ] Model version/architecture shown
- [ ] Intended uses outlined
- [ ] Licenses outlined
- [ ] Ethical considerations addressed
- [ ] Bias considerations addressed
**Status**: 📝 TODO (Week 3)

### 6. Critical Analysis (10 pts)
Answer ONE OR MORE:
- [ ] What is the impact of this project?
- [ ] What does it reveal or suggest?
- [ ] What is the next step?
**Status**: 📝 TODO (Week 2)

### 7. Documentation & Resource Links (5 pts)
- [x] Repo exists (3 pts available)
- [x] README with setup instructions
- [ ] README polish
- [ ] Resource links (2 pts available)
- [ ] Relevant papers cited
- [ ] Code bases cited
**Status**: ✅ MOSTLY DONE (minor polish needed)

### 8. Presentation (10 pts)
- [ ] Organization & Clarity (4 pts)
- [ ] Visual Aids & Demonstrations (3 pts)
- [ ] Delivery & Engagement (2 pts)
- [ ] Preparation & Professionalism (1 pt)
**Status**: 📝 TODO (Week 2)

---

## 🎓 Key Rubric Insights

### What "Methodology" Really Means:
> "Techniques, theories, methodologies or approaches **FROM THE COURSE** have been applied in the solution of the problem presented. **A clear connection has been made to the content.**"

**NOT**: "I built a transformer"
**YES**: "I applied transformer architecture from Week 5 lectures, specifically using multi-head attention (Vaswani et al., covered in class) to capture temporal dependencies in roast profiles. The cross-attention mechanism (discussed in conditional generation lecture) allows the model to condition on bean features..."

### What "Implementation" Really Means:
> "Code **(or pseudocode for a theoretical project)** is discussed and demonstrated."

**NOT**: Perfect, production-ready, fully-optimized code
**YES**: Working demo with explanation, OR well-documented pseudocode

### What "Assessment & Evaluation" Really Means:
> "Approach is assessed and evaluation is explained."

**NOT**: Just showing metrics
**YES**: Explaining why you chose these metrics, what they reveal, what the trade-offs are, how you'd evaluate differently with more resources

---

## 📊 Point Allocation Strategy

### High ROI (Do First):
1. **Methodology writeup** - 50 pts / ~8 hours = 6.25 pts/hour
2. **Basic demo** - 20 pts / ~4 hours = 5 pts/hour
3. **Evaluation framework** - 15 pts / ~3 hours = 5 pts/hour

### Medium ROI (Do Second):
4. **Critical analysis** - 10 pts / ~3 hours = 3.3 pts/hour
5. **Presentation** - 10 pts / ~6 hours = 1.67 pts/hour

### Low ROI (Do Last):
6. **Model card** - 5 pts / ~1 hour = 5 pts/hour (but low priority)
7. **Documentation polish** - 5 pts / ~2 hours = 2.5 pts/hour

### Already Done:
8. **Problem statement** - 10 pts / 0 hours = ✓ free points

---

## ⚠️ Risk Management

### High-Risk Items (Could Lose Many Points):
1. **Methodology too technical, not theoretical** (50 pts at risk)
   - Mitigation: Explicit course connections, cite lectures
2. **No working demo** (20 pts at risk)
   - Mitigation: Simple baseline is enough, pseudocode acceptable
3. **Poor presentation** (10 pts at risk)
   - Mitigation: Practice, visual aids, backup plan

### Medium-Risk Items:
4. **Weak evaluation explanation** (15 pts at risk)
   - Mitigation: Focus on methodology, not just results
5. **Missing critical analysis** (10 pts at risk)
   - Mitigation: Draft early, Week 2

### Low-Risk Items:
6. **Model card** (5 pts) - Easy to complete
7. **Documentation** (5 pts) - Already mostly done
8. **Problem statement** (10 pts) - Already strong

---

## 🔄 Continuous Tracking

### Week 1 Daily Goals:
- **Mon**: Methodology framework (25/50 pts progress)
- **Tue**: Methodology complete (50/50 pts secured)
- **Wed**: Colab training working (20/20 pts secured)
- **Thu**: Generate examples, basic metrics
- **Fri**: Evaluation writeup (15/15 pts secured)

### Progress Tracking:
- [ ] 50/125 pts (40%) - Methodology
- [ ] 70/125 pts (56%) - + Implementation
- [ ] 85/125 pts (68%) - + Evaluation
- [ ] 95/125 pts (76%) - + Critical Analysis
- [ ] 105/125 pts (84%) - + Presentation
- [ ] 110/125 pts (88%) - + Documentation
- [ ] 115/125 pts (92%) - + Model Card

**Target**: 110-120 pts (88-96%)

---

## 💡 Quick Reference: What Matters Most

### Top 3 Priorities (75 pts = 60% of grade):
1. **Methodology connections** (50 pts)
2. **Basic working demo** (20 pts)
3. **Evaluation explanation** (15 pts)

### Everything Else (50 pts = 40% of grade):
4. Presentation (10 pts)
5. Problem statement (10 pts) ✅ done
6. Critical analysis (10 pts)
7. Documentation (5 pts)
8. Model card (5 pts)

---

## 🎯 Success Criteria

### Minimum Viable Capstone (85 pts = B):
- Basic methodology writeup
- Simple working demo
- Evaluation explanation
- Problem statement (already done)

### Target Capstone (110 pts = A):
- Comprehensive methodology with course connections
- Working demo with examples
- Thorough evaluation framework
- Good presentation materials
- Critical analysis
- Polish documentation

### Stretch Capstone (120 pts = A+):
- Exceptional methodology depth
- Impressive visual demo
- Insightful critical analysis
- Professional presentation
- Complete documentation

---

**Remember**: The rubric rewards **understanding and communication** over **implementation perfection**.

A simple model with excellent explanation beats a complex model with weak theory.

---

*Last Updated: Nov 11, 2025*
*Reference this document daily to stay on track!*
