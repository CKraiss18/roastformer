"""
Quick Generation Test: Verify model produces varying temps (not constants)
"""
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from src.dataset.preprocessed_data_loader_NORMALIZED import PreprocessedDataLoader
from src.model.transformer_adapter import AdaptedConditioningModule, AdaptedRoastFormer

print("="*80)
print("GENERATION TEST - Checking if temps vary")
print("="*80)

# Load data
data_loader = PreprocessedDataLoader()
train_profiles, val_profiles = data_loader.load_data()
_, val_loader = data_loader.create_dataloaders(batch_size=1, max_sequence_length=800)

# Load checkpoint
checkpoint_path = Path('checkpoints/test_fix/best_transformer_model.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize model
feature_dims = data_loader.get_feature_dimensions()
conditioning_module = AdaptedConditioningModule(
    num_origins=feature_dims['num_origins'],
    num_processes=feature_dims['num_processes'],
    num_roast_levels=feature_dims['num_roast_levels'],
    num_varieties=feature_dims['num_varieties'],
    num_flavors=feature_dims['num_flavors'],
    embed_dim=32
)

model = AdaptedRoastFormer(
    conditioning_module=conditioning_module,
    d_model=32,
    nhead=2,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    positional_encoding='sinusoidal',
    max_seq_len=800
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get one validation sample
batch = next(iter(val_loader))
features = {
    'categorical': {k: v for k, v in batch['features']['categorical'].items()},
    'continuous': {k: v for k, v in batch['features']['continuous'].items()},
    'flavors': batch['features']['flavors']
}

# Generate 50 steps
start_temp = 426.0
generated = model.generate(features, start_temp=start_temp, target_duration=50, device='cpu')

print(f"\nGenerated first 20 temps:")
print(f"Start: {start_temp:.1f}°F")
for i in range(20):
    print(f"  Step {i+1:2d}: {generated[i]:.1f}°F")

# Check variance
variance = np.var(generated)
temp_range = generated.max() - generated.min()
unique_temps = len(np.unique(np.round(generated, 1)))

print(f"\nStatistics:")
print(f"  Variance: {variance:.2f}")
print(f"  Range: {temp_range:.1f}°F ({generated.min():.1f} - {generated.max():.1f})")
print(f"  Unique values (rounded): {unique_temps}/{len(generated)}")

print(f"\n{'='*80}")
if variance < 1:
    print("❌ FAILED - Constant output (model collapsed)")
elif variance < 100:
    print("⚠️  WARNING - Low variation (weak generation)")
elif generated.min() < 100 or generated.max() > 500:
    print("⚠️  WARNING - Temps outside reasonable range")
elif unique_temps < 10:
    print("⚠️  WARNING - Too few unique values (quantized?)")
else:
    print("✅ SUCCESS - Temps are varying properly!")
    print("   Model generates diverse, reasonable temperatures")
print("="*80)
