# Fixed Cell 10 for Evaluation Notebook

**Issue**: Parameter name mismatches between notebook and model constructor

**Fix**: Replace Cell 10 with this corrected version:

---

```python
# Initialize model from checkpoint
from src.model.transformer_adapter import AdaptedConditioningModule, AdaptedRoastFormer
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

print("Initializing model from checkpoint...")

# Load data to get feature dimensions
data_loader = PreprocessedDataLoader(preprocessed_dir='preprocessed_data')
train_profiles, train_metadata, val_profiles, val_metadata = data_loader.load_all_data()

# Get feature dimensions from checkpoint
feature_dims = checkpoint['feature_dims']
config = checkpoint['config']

print(f"\nFeature dimensions:")
print(f"  Origins: {feature_dims['num_origins']}")
print(f"  Processes: {feature_dims['num_processes']}")
print(f"  Varieties: {feature_dims['num_varieties']}")
print(f"  Flavors: {feature_dims['num_flavors']}")
print(f"  Continuous: {feature_dims['num_continuous']}")

# Initialize conditioning module
conditioning_module = AdaptedConditioningModule(
    num_origins=feature_dims['num_origins'],
    num_processes=feature_dims['num_processes'],
    num_varieties=feature_dims['num_varieties'],
    num_flavors=feature_dims['num_flavors'],
    embed_dim=config['embed_dim'],
    num_continuous=feature_dims['num_continuous']
)

print(f"\n✅ Conditioning module initialized")
print(f"   Condition dim: {conditioning_module.condition_dim}")

# Initialize model
# ⚠️ FIXED PARAMETER NAMES:
#   - positional_encoding_type → positional_encoding
#   - max_seq_length → max_seq_len
model = AdaptedRoastFormer(
    conditioning_module=conditioning_module,
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    dim_feedforward=config['dim_feedforward'],
    dropout=config['dropout'],
    positional_encoding=config['positional_encoding'],  # ✅ FIXED
    max_seq_len=config['max_sequence_length']  # ✅ FIXED
)

print(f"✅ Model architecture initialized")

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model weights loaded successfully")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Device: {device}")
```

---

## Changes Made:

1. **Fixed**: `positional_encoding_type` → `positional_encoding`
2. **Fixed**: `max_seq_length` → `max_seq_len`
3. **Added**: Debug output for feature dimensions
4. **Added**: More detailed success messages

---

## Why This Happened:

The notebook template had placeholder parameter names that didn't match the actual model constructor in `transformer_adapter.py`.

---

**After applying this fix, Cell 10 should run successfully!**
