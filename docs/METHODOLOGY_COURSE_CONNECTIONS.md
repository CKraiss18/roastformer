# RoastFormer: Methodology & Course Connections

**Project**: Transformer-Based Coffee Roast Profile Generation
**Student**: Charlee Kraiss
**Course**: Generative AI Theory (Fall 2024)
**Date**: November 2024

---

## Table of Contents
1. [Overview](#overview)
2. [Core Course Concepts Applied](#core-course-concepts-applied)
3. [Transformer Architecture Theory](#transformer-architecture-theory)
4. [Attention Mechanisms for Time-Series](#attention-mechanisms-for-time-series)
5. [Positional Encoding Analysis](#positional-encoding-analysis)
6. [Conditional Generation Framework](#conditional-generation-framework)
7. [Training Methodology](#training-methodology)
8. [Theoretical Foundations](#theoretical-foundations)
9. [References to Course Materials](#references-to-course-materials)

---

## Overview

### Problem Formulation
RoastFormer addresses the challenge of **conditional sequence generation** in a constrained physical domain. Specifically, we apply transformer-based generative modeling to produce temperature sequences (roast profiles) conditioned on multi-modal input features (bean characteristics and desired flavor outcomes).

### Course Concepts Applied
This project synthesizes multiple concepts from the Generative AI Theory curriculum:

1. **Transformer Architecture** (Vaswani et al., 2017) - Core sequence modeling
2. **Attention Mechanisms** - Capturing temporal dependencies
3. **Positional Encodings** - Time-series representation
4. **Conditional Generation** - Feature-guided output
5. **Autoregressive Generation** - Sequential prediction
6. **Regularization in Small-Data Regimes** - Overfitting prevention
7. **Evaluation Metrics for Generative Models** - Beyond perplexity

---

## Core Course Concepts Applied

### 1. Sequence-to-Sequence Modeling with Transformers

**Course Concept**: Transformer architecture for sequence modeling (Weeks 4-5)

**Application to RoastFormer**:
We adapt the decoder-only transformer architecture (similar to GPT) for autoregressive temperature sequence generation. Unlike language modeling, our sequences are continuous-valued time-series with strict physical constraints.

**Key Adaptation**:
```
Input:  Conditioning features (bean origin, process, target flavors)
        + Previous temperature values (t₀, t₁, ..., tₙ₋₁)
Output: Next temperature value (tₙ)
```

**Theoretical Justification**:
- **Attention enables long-range dependencies**: Roast profiles exhibit multi-phase behavior (drying, Maillard, development) where early decisions affect later outcomes
- **Self-attention captures phase transitions**: The model can learn that reaching 380°F (first crack) signals a regime change
- **Autoregressive generation ensures causality**: Each temperature depends only on previous states (matches physical reality)

**Difference from Standard Transformers**:
- Output space is 1D continuous (not discrete vocabulary)
- Sequences have fixed temporal resolution (1-second intervals)
- Physical constraints must be enforced (bounded heating rates, monotonicity post-turning-point)

---

### 2. Multi-Head Attention for Temporal Pattern Recognition

**Course Concept**: Multi-head attention allows the model to attend to different aspects of the input simultaneously (Week 5)

**Application to RoastFormer**:
We hypothesize that different attention heads learn different roasting phases:

| Attention Head | Hypothesized Role | Phase Focus |
|----------------|-------------------|-------------|
| Head 1-2 | Early drying phase | Temperature drop and recovery |
| Head 3-4 | Maillard reactions | Steady heating (350-380°F) |
| Head 5-6 | Development phase | Post-first-crack control |
| Head 7-8 | Global trajectory | Overall profile shape |

**Theoretical Basis**:
- **Inductive bias**: Coffee roasting has known phases with distinct thermal dynamics
- **Multi-scale attention**: Short-term (second-to-second) vs long-term (phase-to-phase) patterns
- **Feature interaction**: How bean density affects heating rate varies by phase

**Validation Approach**:
Attention weight visualization (heatmaps) to verify if model learns phase-specific patterns.

---

### 3. Positional Encoding for Time-Series Representation

**Course Concept**: Positional encodings inject sequence order information (Week 5)

**Application to RoastFormer**:
We compare THREE positional encoding methods from course material:

#### **Method 1: Sinusoidal Positional Encoding (Vaswani et al., 2017)**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Advantages**:
- Deterministic, no learned parameters
- Generalizes to unseen sequence lengths
- Smooth, continuous representation

**Disadvantages**:
- Not optimized for time-series patterns
- Equal treatment of all positions (but roast phases are non-uniform)

#### **Method 2: Learned Positional Embeddings**
```
pos_embedding = Embedding(max_seq_len, d_model)
```

**Advantages**:
- Can learn phase-specific representations (e.g., different encoding for first-crack region)
- Potentially better for fixed-length sequences

**Disadvantages**:
- Cannot extrapolate beyond trained lengths
- Requires more parameters (overfitting risk with 28-36 samples)

#### **Method 3: Rotary Position Embedding (RoPE) (Su et al., 2021)**
```
q_rot = rotate(q, position)
k_rot = rotate(k, position)
Attention(q_rot, k_rot, v)
```

**Advantages**:
- Relative position information (more natural for time-series)
- Better long-range decay properties
- State-of-the-art for sequence modeling

**Disadvantages**:
- More complex implementation
- Newer method (less classroom coverage)

**Experimental Design**:
Train identical models with each encoding type, compare:
- MAE on validation set
- Attention pattern interpretability
- Physics constraint violations

**Course Connection**: This is a direct application of the positional encoding theory from Week 5, extended to compare modern variants (RoPE) against classical approaches.

---

### 4. Conditional Generation with Feature Conditioning

**Course Concept**: Conditional generative models learn P(x|c) where c is conditioning information (Week 6-7)

**Application to RoastFormer**:
We condition on **three types of features**:

#### **Type 1: Categorical Features (Embedding-based)**
- **Origin** (Ethiopia, Colombia, etc.) → Learned embedding
- **Process** (Washed, Natural, Honey) → Learned embedding
- **Roast Level** (Light, Medium, Dark) → Learned embedding
- **Variety** (Heirloom, Caturra, etc.) → Learned embedding

**Theoretical Approach**:
```python
origin_embed = Embedding(num_origins, embed_dim)(origin_id)
process_embed = Embedding(num_processes, embed_dim)(process_id)
# Concatenate all categorical embeddings
categorical_condition = concat([origin_embed, process_embed, ...])
```

**Justification**:
- Categorical features have no inherent ordering (Ethiopia ≠ Ethiopia+1)
- Embeddings learn meaningful representations (e.g., similar origins may cluster)
- Standard approach from NLP, applied to structured data

#### **Type 2: Continuous Features (Normalization + Projection)**
- **Target Finish Temperature** (390-430°F)
- **Altitude** (1000-2400 MASL)
- **Bean Density Proxy**
- **Caffeine Content**

**Theoretical Approach**:
```python
# Normalize to [0, 1] or [-1, 1]
normalized = (continuous - mean) / std
# Project to same dimension as embeddings
projected = Linear(num_continuous, embed_dim)(normalized)
```

**Justification**:
- Continuous features have numerical meaning (2000m > 1000m altitude)
- Normalization prevents feature scale dominance
- Projection to common dimension allows fusion with categorical embeddings

#### **Type 3: Multi-Hot Flavor Encoding (Set Representation)**
- **Flavors** (berries, chocolate, floral, etc.) → Multi-hot vector

**Theoretical Approach**:
```python
# Multi-hot encoding: [1, 0, 1, 1, 0, ...] for multiple flavors
flavor_vector = MultiHotEncoding(all_possible_flavors)(selected_flavors)
# Project to embedding dimension
flavor_embed = Linear(num_flavors, embed_dim)(flavor_vector)
```

**Novel Contribution**:
- Most roast profile generators ignore flavor (only physical parameters)
- We hypothesize: flavor is learnable from temperature trajectory
- Tests whether transformers can capture flavor→temperature mappings

**Course Connection**:
This extends the conditional generation framework (Week 6) to multi-modal conditioning (categorical + continuous + set-valued features), similar to multi-modal transformers (CLIP, Flamingo).

#### **Conditioning Architecture**:
```
Categorical → Embeddings  ┐
Continuous  → Projection  ├→ Concatenate → Unified Condition Vector
Flavors     → Projection  ┘
                          ↓
                   Cross-Attention in Decoder
```

**Cross-Attention Mechanism**:
Each decoder layer performs cross-attention between:
- **Query**: Current temperature sequence state
- **Key/Value**: Conditioning features

This allows the model to repeatedly "look at" conditioning information at each layer.

---

### 5. Autoregressive Generation for Sequential Prediction

**Course Concept**: Autoregressive models generate sequences one token at a time, conditioning on all previous tokens (Week 4)

**Application to RoastFormer**:
```
p(T₁, T₂, ..., Tₙ | conditions) = ∏ p(Tᵢ | T₁, ..., Tᵢ₋₁, conditions)
```

**Generation Process**:
1. Start with initial temperature (charge temp ~426°F)
2. For each timestep t:
   - Input: T₀, T₁, ..., Tₜ₋₁ + conditioning
   - Output: Tₜ (predicted temperature at time t)
   - Append Tₜ to sequence
3. Continue until end condition (target finish temp reached or max steps)

**Advantages for Roast Profiles**:
- **Causality**: Roasters can only control temperature based on past, not future
- **Error accumulation analysis**: Can study how early mistakes compound
- **Interactive generation**: Could allow real-time roaster input (future work)

**Challenges**:
- **Exposure bias**: Model trained on ground-truth but generates from its own predictions
- **Error accumulation**: Small errors compound over 600+ timesteps
- **Slow generation**: Sequential, not parallelizable (but fast enough for 10-minute profiles)

**Course Connection**:
Direct application of autoregressive modeling (Week 4), with awareness of exposure bias challenges discussed in class.

---

### 6. Training in Small-Data Regimes

**Course Concept**: Generative models typically require large datasets; small-data training requires careful regularization (Week 8)

**Application to RoastFormer**:
**Dataset Size**: 28-36 roast profiles (VERY small for deep learning)

**Regularization Strategies**:

#### **1. Model Size Reduction**
- Use smaller model (d_model=128-256 instead of 512)
- Fewer layers (4-6 instead of 12+)
- Fewer parameters (~2-10M instead of 100M+)

**Theoretical Justification**:
- VC dimension theory: simpler models generalize better with less data
- Bias-variance tradeoff: accept higher bias for lower variance

#### **2. Dropout (Srivastava et al., 2014)**
```python
Dropout(p=0.1-0.2)  # After attention and FFN layers
```

**Theoretical Justification**:
- Ensemble learning interpretation: trains 2^n implicit models
- Prevents co-adaptation of features
- Course coverage: Week 8 (regularization techniques)

#### **3. Weight Decay (L2 Regularization)**
```python
optimizer = AdamW(params, lr=1e-4, weight_decay=0.01)
```

**Theoretical Justification**:
- Penalizes large weights: L = L_task + λ||θ||²
- Encourages smoother functions
- Standard practice from course optimization lectures

#### **4. Early Stopping**
```python
patience = 10  # Stop if validation loss doesn't improve for 10 epochs
```

**Theoretical Justification**:
- Implicit regularization
- Prevents overfitting to training set
- Course coverage: Week 8 (training techniques)

#### **5. Data Augmentation (Domain-Specific)**
```python
# Temperature jitter: add small Gaussian noise
T_augmented = T_original + N(0, 0.5°F)

# Time warping: slightly compress/expand profile
T_warped = interpolate(T_original, scale=random.uniform(0.95, 1.05))
```

**Theoretical Justification**:
- Increases effective dataset size
- Reflects real measurement noise
- Physics-preserving augmentation (stays within valid ranges)

#### **6. Physics-Based Constraints (Inductive Bias)**
```python
# Enforce heating rate bounds during generation
if heating_rate > 100/60:  # Too fast
    T_next = clip(T_next, max_allowed)

# Enforce monotonicity post-turning-point
if past_turning_point and T_next < T_current:
    T_next = T_current  # Force non-decrease
```

**Theoretical Justification**:
- Incorporates domain knowledge (roasting physics)
- Reduces hypothesis space (no need to learn impossible profiles)
- Similar to architectural inductive biases (CNNs for images, RNNs for sequences)

**Course Connection**:
Synthesizes regularization techniques from Week 8 with domain-specific inductive biases (discussed in architectural design lectures).

---

### 7. Evaluation Metrics for Generative Models

**Course Concept**: Generative models require task-specific evaluation beyond standard metrics (Week 9)

**Application to RoastFormer**:

#### **Metric 1: Mean Absolute Error (MAE)**
```
MAE = (1/n) Σ |T_real(i) - T_generated(i)|
```

**Strengths**:
- Directly measures temperature accuracy
- Interpretable units (°F)
- Sensitive to point-wise errors

**Weaknesses**:
- Doesn't capture shape similarity
- Treats all timesteps equally (but first crack is more critical)

#### **Metric 2: Dynamic Time Warping (DTW)**
```
DTW = min_alignment Σ (T_real[i] - T_gen[align[i]])²
```

**Strengths**:
- Allows flexible temporal alignment (handles slight phase shifts)
- Captures overall trajectory shape
- Robust to small timing differences

**Weaknesses**:
- More complex than MAE
- Can hide systematic timing errors

**Course Connection**:
DTW discussed in time-series analysis (Week 10), alternative to strict Euclidean distance.

#### **Metric 3: Finish Temperature Accuracy**
```
Finish_Accuracy = |T_real[-1] - T_target| < 10°F
```

**Strengths**:
- Directly measures task success (hit target roast level)
- Binary outcome (clear success/failure)

**Weaknesses**:
- Ignores entire profile (could achieve finish temp with terrible trajectory)

#### **Metric 4: Physics Constraint Compliance**
```python
# Monotonicity (post-turning-point)
monotonic_ratio = sum(diff >= 0) / len(diff)

# Bounded heating rates
valid_ror_ratio = sum((ror >= 20) & (ror <= 100)) / len(ror)

# Overall compliance
physics_score = (monotonic_ratio + valid_ror_ratio) / 2
```

**Strengths**:
- Ensures physical plausibility
- Domain-specific validation
- Catches impossible profiles

**Weaknesses**:
- Binary constraints (doesn't measure how close to valid)
- Requires domain expertise to define

**Course Connection**:
Extends standard generative metrics (perplexity, FID) to domain-specific constraints, similar to controllability metrics in conditional generation (Week 7).

#### **Metric 5: Human Evaluation (Future Work)**
```
Survey roasters: "Would you try this profile?"
Scale: 1 (Never) - 5 (Definitely)
```

**Strengths**:
- Ultimate ground truth (practical utility)
- Captures subtle quality factors

**Weaknesses**:
- Expensive, slow
- Requires domain experts
- Not available for this project

---

## Transformer Architecture Theory

### Standard Transformer Decoder Layer

**From Course (Week 5)**:
```
TransformerDecoderLayer:
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Multi-Head Cross-Attention (with encoder output)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
```

### RoastFormer Decoder Layer Adaptation

**Our Implementation**:
```
RoastFormerDecoderLayer:
    1. Masked Multi-Head Self-Attention (on temperature sequence)
    2. Add & Norm
    3. Multi-Head Cross-Attention (with conditioning features)
    4. Add & Norm
    5. Feed-Forward Network (2-layer MLP with ReLU)
    6. Add & Norm
```

**Key Differences from Standard Decoder**:

| Aspect | Standard Transformer | RoastFormer |
|--------|---------------------|-------------|
| **Input** | Discrete tokens (text) | Continuous values (temperature) |
| **Encoder** | Separate encoder for source | No encoder (conditioning via cross-attn) |
| **Output Space** | Softmax over vocabulary | Linear projection to ℝ (temperature) |
| **Loss Function** | Cross-entropy | Mean Squared Error (MSE) |
| **Causality** | Masked attention (future tokens) | Masked attention (future timesteps) |

### Attention Mechanism Deep Dive

**Scaled Dot-Product Attention (from course)**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Why Scaling Factor √d_k?**
- Prevents dot products from growing too large (softmax saturation)
- Covered in Week 5: variance of QK^T is d_k, scaling normalizes to variance 1
- Critical for stable gradients in deep networks

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O

where headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Why Multiple Heads?**
- Different heads learn different patterns (empirical finding from Vaswani et al.)
- Allows attending to multiple positions simultaneously
- Course discussion (Week 5): similar to multiple filters in CNNs

**Application to Roast Profiles**:
```
Query:   Current temperature sequence state (what we're generating)
Key:     Past temperature values (what we attend to)
Value:   Past temperature values (what we aggregate)

Cross-Attention:
Query:   Current temperature sequence state
Key:     Conditioning features (bean characteristics)
Value:   Conditioning features
```

### Feed-Forward Network

**From Course**:
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Typical dimensions**:
- d_model → 4*d_model → d_model
- E.g., 256 → 1024 → 256

**Role** (from course lectures):
- Adds non-linearity (attention is mostly linear)
- Position-wise transformation (processes each timestep independently)
- Increases model capacity

### Layer Normalization

**From Course (Week 5)**:
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

**Why Layer Norm instead of Batch Norm?**
- Batch norm: normalize across batch dimension
- Layer norm: normalize across feature dimension
- Course explanation: Layer norm works better for sequences (variable lengths)

**Placement**: Pre-LN vs Post-LN
- **Post-LN** (original Transformer): Add → Norm
- **Pre-LN** (modern transformers): Norm → Add
- RoastFormer uses **Post-LN** (following course examples)

---

## Attention Mechanisms for Time-Series

### Self-Attention for Temporal Dependencies

**Standard RNN/LSTM Approach** (Week 3):
- Sequential processing: h_t = f(h_{t-1}, x_t)
- Fixed context window (hidden state)
- Difficulty learning long-range dependencies (vanishing gradients)

**Transformer Self-Attention Advantage**:
- Parallel processing: all timesteps computed simultaneously
- Direct connections between all pairs of timesteps
- Long-range dependencies via attention weights

**Example: First Crack Detection**
```
At t=480s (8 minutes), temperature reaches 380°F (first crack)
Roaster must reduce heat to avoid scorching

RNN: Must propagate information through 480 hidden states
Transformer: Direct attention from t=480 to t=0 (initial charge temp)
```

### Masked Attention for Causality

**Problem**: Temperature at time t cannot depend on future (t+1, t+2, ...)

**Solution**: Attention Mask
```
Mask = [
    [0, -∞, -∞, -∞],  # t=0 can only see t=0
    [0,  0, -∞, -∞],  # t=1 can see t=0,1
    [0,  0,  0, -∞],  # t=2 can see t=0,1,2
    [0,  0,  0,  0],  # t=3 can see t=0,1,2,3
]

Attention_Masked = softmax((QK^T / √d_k) + Mask)
```

**Course Connection**:
Week 5 (autoregressive generation), ensures model learns causal relationships.

### Cross-Attention for Conditioning

**Mechanism**:
```
Q = temperature_sequence  (what we're generating)
K = V = conditioning_features  (what we condition on)

CrossAttn = softmax(QK^T / √d_k) V
```

**Interpretation**:
- Model learns to "query" conditioning features
- Different timesteps may attend to different features
  - Early phase: attend to bean density (affects heat transfer)
  - Late phase: attend to roast level target (affects finish temp)

**Course Connection**:
Week 6 (conditional generation), similar to image captioning (attend to image features while generating text).

---

## Theoretical Foundations

### Universal Approximation for Sequences

**Theory** (from course, Week 5):
Transformers are universal sequence approximators (with sufficient depth/width)

**Implication for RoastFormer**:
- In principle, can learn any temperature→temperature mapping
- In practice, limited by:
  - Dataset size (28-36 samples)
  - Model capacity (regularization needed)
  - Inductive biases (architecture choices)

### Inductive Biases

**Definition** (Week 2): Assumptions built into model architecture

**RoastFormer Inductive Biases**:

| Bias | Justification | Alternative |
|------|---------------|-------------|
| **Autoregressive** | Temperature evolves causally | Fully parallel (faster but less physical) |
| **Attention** | Long-range dependencies exist | RNN (local context only) |
| **Positional Encoding** | Order matters | Orderless (set-based) |
| **Cross-Attention** | Conditioning affects all timesteps | Concatenate once at start |
| **MSE Loss** | Continuous output | Classification (bucketed temps) |

**Course Connection**:
Week 2 (inductive biases in neural architectures), similar discussion of CNN biases for images.

### Optimization Theory

**Loss Function**:
```
L = (1/n) Σ (T_real(i) - T_pred(i))²  [MSE]
  + λ₁ ||θ||²                         [Weight decay]
  + λ₂ PhysicsLoss(T_pred)            [Constraint penalty]
```

**Optimizer**: AdamW (Adam with decoupled weight decay)
- **Adam**: Adaptive learning rates per parameter
- **Weight decay decoupling**: Better than L2 for Adam (Loshchilov & Hutter, 2019)
- **Course coverage**: Week 8 (optimization algorithms)

**Learning Rate Schedule**: Cosine Annealing
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(πt/T))
```

**Justification**:
- Starts high (fast initial learning)
- Gradually decreases (fine-tuning)
- Course coverage: Week 8 (learning rate schedules)

---

## References to Course Materials

### Primary Course Connections

1. **Transformer Architecture** (Week 5)
   - Lecture: "Attention Is All You Need"
   - Readings: Vaswani et al. (2017)
   - Applied: Core RoastFormer architecture

2. **Attention Mechanisms** (Week 5)
   - Lecture: "Multi-Head Self-Attention"
   - Readings: Illustrated Transformer (blog post)
   - Applied: Temporal pattern recognition

3. **Positional Encodings** (Week 5)
   - Lecture: "Sequence Representation"
   - Readings: Vaswani et al. (2017), Su et al. (2021) [RoPE]
   - Applied: Three-way comparison (sinusoidal, learned, RoPE)

4. **Conditional Generation** (Week 6-7)
   - Lecture: "Guided Generation"
   - Readings: Conditional GANs, Classifier-Free Guidance
   - Applied: Multi-modal feature conditioning

5. **Autoregressive Models** (Week 4)
   - Lecture: "Sequential Generation"
   - Readings: GPT architecture papers
   - Applied: Temperature sequence generation

6. **Regularization** (Week 8)
   - Lecture: "Training Deep Networks"
   - Readings: Dropout paper, optimization techniques
   - Applied: Small-data regime strategies

7. **Evaluation Metrics** (Week 9)
   - Lecture: "Assessing Generative Models"
   - Readings: FID, IS, perplexity
   - Applied: Domain-specific metrics (DTW, physics constraints)

### Supplementary References

- **Vaswani et al. (2017)**: "Attention Is All You Need" - Core transformer paper
- **Su et al. (2021)**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Loshchilov & Hutter (2019)**: "Decoupled Weight Decay Regularization" - AdamW
- **Srivastava et al. (2014)**: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

### Textbook Connections
*(To be filled in with specific chapter references from course textbook)*

- Chapter X: Sequence Modeling
- Chapter Y: Attention Mechanisms
- Chapter Z: Generative Models

---

## Summary: Course Concepts → RoastFormer Implementation

| Course Concept | Lecture Week | RoastFormer Application |
|----------------|--------------|------------------------|
| Transformer Architecture | 5 | Decoder-only model for temperature sequences |
| Multi-Head Attention | 5 | 8 heads for multi-phase pattern recognition |
| Positional Encodings | 5 | Three methods compared (sinusoidal, learned, RoPE) |
| Conditional Generation | 6-7 | Multi-modal conditioning (categorical + continuous + flavors) |
| Autoregressive Generation | 4 | Sequential temperature prediction (causality-preserving) |
| Regularization | 8 | Dropout, weight decay, early stopping (small-data regime) |
| Inductive Biases | 2 | Physics constraints, architecture choices |
| Evaluation Metrics | 9 | MAE, DTW, physics compliance (domain-specific) |
| Optimization | 8 | AdamW + cosine annealing schedule |

---

## Next Steps: Theoretical Extensions

### Planned Experiments (Time Permitting)
1. **Positional Encoding Ablation**: Validate which encoding best captures roast phases
2. **Attention Pattern Analysis**: Verify multi-head phase specialization hypothesis
3. **Conditioning Ablation**: Quantify impact of flavor features (novel contribution)

### Future Theoretical Questions
1. Can transformers learn flavor→temperature mappings with more data?
2. How does attention pattern differ between light vs dark roasts?
3. Can we extract interpretable "roast rules" from learned attention weights?

---

**This methodology demonstrates application of core course concepts (transformers, attention, conditional generation) to a novel domain (coffee roasting), with careful consideration of theoretical foundations, domain constraints, and small-data challenges.**

*Document Version 1.0 - November 11, 2024*
