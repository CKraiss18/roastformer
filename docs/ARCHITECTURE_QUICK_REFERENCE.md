# RoastFormer Architecture Reference - Quick Guide

## 📚 **What's Inside**

Complete transformer architecture code for RoastFormer, consolidated from all previous discussions.

### **File:** `ROASTFORMER_ARCHITECTURE_REFERENCE.py`

---

## 📋 **Sections Overview**

### **Section 1: Feature Encoding & Conditioning**
- `FeatureEncoder` class
  - Encodes categorical features (origin, process, variety, roast level)
  - Encodes continuous features (temp, altitude, density, caffeine)
  - **Flavor encoding:** One-hot AND embedding approaches
  - Builds flavor vocabulary from dataset

### **Section 2: Conditioning Module**
- `ConditioningModule` class
  - Combines all features into unified conditioning vector
  - Categorical embeddings (origin, process, variety, roast level, flavors)
  - Continuous feature projection
  - Outputs single conditioning vector for transformer

### **Section 3: Positional Encoding**
Three implementations for ablation study:
- `PositionalEncoding` - Standard sinusoidal (Vaswani et al.)
- `LearnedPositionalEncoding` - Learned embeddings
- `RotaryPositionalEncoding` - RoPE (modern, more stable)

### **Section 4: RoastFormer Architecture**
- `RoastFormer` class - Main decoder-only transformer
  - Conditioning-aware architecture
  - Teacher forcing for training
  - Autoregressive generation for inference
  - Configurable: d_model, nhead, num_layers, positional encoding

### **Section 5: Training Utilities**
- `RoastProfileDataset` - PyTorch Dataset wrapper
- `train_roastformer()` - Complete training loop
  - AdamW optimizer
  - MSE loss
  - Gradient clipping
  - Validation monitoring

### **Section 6: Usage Example**
- Complete end-to-end example
- Load dataset → Initialize encoders → Create model → Generate profile
- Includes visualization code

---

## 🎯 **Key Features**

### **User Prompt to Generation**
```python
# User provides:
prompt = {
    'origin': 'Ethiopia',
    'process': 'Washed', 
    'variety': 'Heirloom',
    'roast_level': 'Light',
    'target_finish_temp': 395,
    'altitude': 2000,
    'flavors': ['berries', 'floral', 'citrus']  # ← Flavor conditioning!
}

# Model generates:
generated_profile = model.generate(
    categorical_indices=encode_categorical(prompt),
    continuous_features=encode_continuous(prompt),
    flavor_notes=prompt['flavors'],
    start_temp=426.0,
    target_duration=600
)
```

### **Flavor Conditioning (Two Approaches)**

**Approach 1: One-Hot Encoding**
```python
# Flavor vocabulary: ['berries', 'chocolate', 'citrus', 'floral', ...]
# Input: ['berries', 'floral']
# One-hot: [1, 0, 0, 1, 0, 0, ...]
# Project to embed_dim via Linear layer
```

**Approach 2: Embedding Layer**
```python
# Each flavor gets learned embedding
# Input: ['berries', 'floral']
# Embeddings: [emb_berries, emb_floral]
# Average: (emb_berries + emb_floral) / 2
```

### **Positional Encoding Variants**
For your ablation study (proposal: "test 3 positional encoding variants"):
1. **Sinusoidal** - Standard, no learned params
2. **Learned** - Separate embedding per position
3. **Rotary (RoPE)** - Modern, rotation-based

---

## 📊 **Model Configuration**

### **Recommended Settings**

**Small Model (Fast Training):**
```python
model = RoastFormer(
    conditioning_module=conditioning_module,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=512,
    positional_encoding='sinusoidal'
)
# ~2M parameters
```

**Medium Model (Balanced):**
```python
model = RoastFormer(
    conditioning_module=conditioning_module,
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
    positional_encoding='sinusoidal'
)
# ~10M parameters
```

**Large Model (High Quality):**
```python
model = RoastFormer(
    conditioning_module=conditioning_module,
    d_model=512,
    nhead=8,
    num_layers=8,
    dim_feedforward=2048,
    positional_encoding='rotary'
)
# ~40M parameters
```

---

## 🚀 **Development Roadmap**

### **Phase 1: Baseline (Nov 3-5)**
- Implement basic transformer (Section 4)
- Use Phase 1 features only (origin, process, roast level, finish temp)
- Sinusoidal positional encoding
- Test on Onyx dataset

### **Phase 2: Enhanced Conditioning (Nov 6-8)**
- Add Phase 2 features (variety, altitude, density)
- Implement ConditioningModule (Section 2)
- Test conditioning effectiveness

### **Phase 3: Flavor Conditioning (Nov 9-12)**
- Add flavor feature extraction to dataset
- Implement flavor encoding (Section 1)
- Test flavor-guided generation

### **Phase 4: Ablations (Nov 13-15)**
- Test 3 positional encoding variants (Section 3)
- Compare conditioning strategies
- Analyze attention patterns

---

## 💡 **Usage Patterns**

### **Training**
```python
# 1. Load and encode dataset
df = pd.read_csv('onyx_dataset/dataset_summary.csv')
feature_encoder = FeatureEncoder(df)

# 2. Create dataset and dataloader
dataset = RoastProfileDataset(df, feature_encoder)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Initialize model
conditioning_module = ConditioningModule(feature_encoder, embed_dim=32)
model = RoastFormer(conditioning_module, d_model=256, nhead=8, num_layers=6)

# 4. Train
model = train_roastformer(model, train_loader, val_loader, num_epochs=100)
```

### **Generation**
```python
# Prepare conditioning
categorical = {'origin': 0, 'process': 1, 'variety': 2, 'roast_level': 0}
continuous = torch.tensor([395/425, 2000/2500, 0.75/0.80, 210/230])
flavors = ['berries', 'floral', 'citrus']

# Generate
profile = model.generate(
    categorical_indices=categorical,
    continuous_features=continuous,
    flavor_notes=flavors,
    start_temp=426.0,
    target_duration=600
)
```

### **Validation**
```python
# Compare against real Onyx profile
real_profile = load_onyx_profile('geometry')
generated_profile = model.generate(...)

# Calculate metrics
mae = np.mean(np.abs(real_profile - generated_profile))
dtw_distance = dynamic_time_warping(real_profile, generated_profile)

print(f"MAE: {mae:.2f}°F")
print(f"DTW: {dtw_distance:.2f}")
```

---

## 📈 **Expected Results**

Based on your proposal success criteria:

✅ **MAE <5°F** for temperature predictions  
✅ **100% monotonic** increase post-turning-point  
✅ **>95% bounded** heating rates (20-100°F/min)  
✅ **>90%** reach target finish temps  
✅ **Smooth transitions** (no jumps >10°F/sec)

---

## 🔧 **Customization Points**

### **Easy to Modify:**
1. **Conditioning features** - Add/remove in `ConditioningModule.__init__`
2. **Positional encoding** - Change `positional_encoding` parameter
3. **Model size** - Adjust `d_model`, `nhead`, `num_layers`
4. **Flavor encoding** - Switch `use_flavor_embeddings` True/False

### **Advanced Modifications:**
1. **Multi-task learning** - Add RoR prediction head
2. **Attention visualization** - Extract attention weights
3. **Constrained generation** - Add physics-based constraints
4. **Curriculum learning** - Start with easy profiles

---

## 📞 **Next Steps**

1. ✅ **Save this reference** - All transformer code in one place
2. 🔄 **Add flavor extraction** to dataset builder (next task!)
3. 🚀 **Implement baseline** - Start with Phase 1 features
4. 📊 **Validate** - Test against Onyx profiles
5. 🎨 **Add flavors** - Enhance with flavor conditioning

---

## 📦 **File Contents Summary**

**6 Major Sections:**
1. Feature Encoding (categorical, continuous, flavors)
2. Conditioning Module (unified embedding)
3. Positional Encoding (3 variants)
4. RoastFormer Model (decoder-only transformer)
5. Training Utilities (dataset, training loop)
6. Usage Example (end-to-end demo)

**~400 lines** of production-ready code with documentation.

**Everything you need** to implement, train, and deploy RoastFormer! 🎯☕🤖

---

**Ready to move to flavor extraction for the dataset builder!** 🎨
