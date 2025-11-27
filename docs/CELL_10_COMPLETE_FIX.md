# Complete Fixed Cell 10 for Evaluation Notebook

**All Issues Fixed:**
1. ✅ `load_all_data()` → `load_data()`
2. ✅ Use `get_feature_dimensions()` from data loader
3. ✅ `num_continuous` → `num_roast_levels` in conditioning module
4. ✅ `positional_encoding_type` → `positional_encoding`
5. ✅ `max_seq_length` → `max_seq_len`

---

## Replace Cell 10 with this:

```python
# Initialize model from checkpoint
from src.model.transformer_adapter import AdaptedConditioningModule, AdaptedRoastFormer
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

print("="*80)
print("INITIALIZING MODEL FROM CHECKPOINT")
print("="*80)

# Load data to get feature dimensions
data_loader = PreprocessedDataLoader(preprocessed_dir='preprocessed_data')
train_profiles, val_profiles = data_loader.load_data()  # ✅ FIXED: load_data() not load_all_data()

# Get feature dimensions from data loader
feature_dims = data_loader.get_feature_dimensions()  # ✅ FIXED: Use data loader method

print(f"\n📊 Feature Dimensions:")
print(f"   Origins: {feature_dims['num_origins']}")
print(f"   Processes: {feature_dims['num_processes']}")
print(f"   Roast Levels: {feature_dims['num_roast_levels']}")
print(f"   Varieties: {feature_dims['num_varieties']}")
print(f"   Flavors: {feature_dims['num_flavors']}")
print(f"   Continuous: {feature_dims['num_continuous']}")

# Get model config from checkpoint
config = checkpoint['config']

print(f"\n🏗️ Model Configuration:")
print(f"   d_model: {config['d_model']}")
print(f"   nhead: {config['nhead']}")
print(f"   num_layers: {config['num_layers']}")
print(f"   positional_encoding: {config['positional_encoding']}")

# Initialize conditioning module
conditioning_module = AdaptedConditioningModule(
    num_origins=feature_dims['num_origins'],
    num_processes=feature_dims['num_processes'],
    num_roast_levels=feature_dims['num_roast_levels'],  # ✅ FIXED: was num_continuous
    num_varieties=feature_dims['num_varieties'],
    num_flavors=feature_dims['num_flavors'],
    embed_dim=config['embed_dim']
)

print(f"\n✅ Conditioning module initialized")
print(f"   Condition dim: {conditioning_module.condition_dim}")

# Initialize model
model = AdaptedRoastFormer(
    conditioning_module=conditioning_module,
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    dim_feedforward=config['dim_feedforward'],
    dropout=config['dropout'],
    positional_encoding=config['positional_encoding'],  # ✅ FIXED: was positional_encoding_type
    max_seq_len=config['max_sequence_length']  # ✅ FIXED: was max_seq_length
)

print(f"✅ Model architecture initialized")

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model weights loaded successfully")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Device: {device}")

print("="*80)
```

---

## Summary of All Fixes:

| Issue | Wrong | Correct |
|-------|-------|---------|
| Method name | `load_all_data()` | `load_data()` |
| Feature dims source | `checkpoint['feature_dims']` | `data_loader.get_feature_dimensions()` |
| Conditioning param | `num_continuous` | `num_roast_levels` |
| Model param 1 | `positional_encoding_type` | `positional_encoding` |
| Model param 2 | `max_seq_length` | `max_seq_len` |

---

**This should work now!** 🎯
