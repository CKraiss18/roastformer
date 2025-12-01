"""
Generation Test: Tiny Model (d=64)
Expected: Smooth curves, realistic roast profiles
"""
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from src.dataset.preprocessed_data_loader_NORMALIZED import PreprocessedDataLoader
from src.model.transformer_adapter import AdaptedConditioningModule, AdaptedRoastFormer

print("="*80)
print("GENERATION TEST - Tiny Model (d=64)")
print("="*80)

# Load data
data_loader = PreprocessedDataLoader()
train_profiles, val_profiles = data_loader.load_data()
_, val_loader = data_loader.create_dataloaders(batch_size=1, max_sequence_length=800)

# Load checkpoint
checkpoint_path = Path('checkpoints/tiny_normalized/best_transformer_model.pt')
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
    d_model=64,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.2,
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

# Generate full profile (600 steps = 10 minutes)
start_temp = 426.0
generated = model.generate(features, start_temp=start_temp, target_duration=600, device='cpu')

print(f"\nGenerated profile statistics:")
print(f"  Start temp: {generated[0]:.1f}°F")
print(f"  Final temp: {generated[-1]:.1f}°F")
print(f"  Min temp: {generated.min():.1f}°F")
print(f"  Max temp: {generated.max():.1f}°F")
print(f"  Range: {generated.max() - generated.min():.1f}°F")
print(f"  Variance: {np.var(generated):.2f}")

# Check smoothness (rate of change)
ror = np.diff(generated) * 60  # Rate of rise in °F/min
print(f"\nRate of Rise (RoR) statistics:")
print(f"  Mean RoR: {np.mean(ror):.2f}°F/min")
print(f"  Max RoR: {np.max(ror):.2f}°F/min")
print(f"  Min RoR: {np.min(ror):.2f}°F/min")
print(f"  RoR std dev: {np.std(ror):.2f}°F/min")

# Check for sudden jumps (smoothness)
max_jump = np.max(np.abs(np.diff(generated)))
print(f"\nSmoothness:")
print(f"  Max jump: {max_jump:.2f}°F/sec")
print(f"  Smooth transitions: {'✅ Yes' if max_jump < 10 else '⚠️  Some jumps'}")

# Sample points
print(f"\nSample points:")
for i in [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 599]:
    print(f"  t={i:3d}s: {generated[i]:6.1f}°F")

# Overall assessment
print(f"\n{'='*80}")
variance = np.var(generated)
temp_range = generated.max() - generated.min()

if variance > 1000 and temp_range > 50 and max_jump < 20:
    print("✅✅ EXCELLENT!")
    print("   Profile is smooth, realistic, and well-behaved")
    print("   Ready for evaluation and presentation!")
elif variance > 500 and temp_range > 30:
    print("✅ GOOD!")
    print("   Profile shows reasonable variation")
    print("   Minor improvements possible but usable")
elif variance > 100:
    print("⚠️  ACCEPTABLE")
    print("   Profile varies but may need refinement")
else:
    print("❌ NEEDS WORK")
    print("   Profile shows limited variation")

print("="*80)
