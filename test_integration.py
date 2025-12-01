"""
Integration Test for RoastFormer Components

Tests that all components load and work together without errors.
Does not actually train, just validates the pipeline.

Author: Charlee Kraiss
"""

import sys
from pathlib import Path

print("="*80)
print("ROASTFORMER INTEGRATION TEST")
print("="*80)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from src.dataset.preprocessed_data_loader import PreprocessedDataLoader
    from src.model.transformer_adapter import (
        AdaptedConditioningModule,
        AdaptedRoastFormer,
        SinusoidalPositionalEncoding,
        LearnedPositionalEncoding
    )
    import torch
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load preprocessed data
print("\n[2/5] Testing data loader...")
try:
    data_loader = PreprocessedDataLoader(preprocessed_dir="preprocessed_data")
    train_profiles, val_profiles = data_loader.load_data()
    train_loader, val_loader = data_loader.create_dataloaders(batch_size=4, max_sequence_length=800)
    feature_dims = data_loader.get_feature_dimensions()
    print(f"✓ Data loaded: {len(train_profiles)} train, {len(val_profiles)} val")
    print(f"✓ Feature dims: {feature_dims}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 3: Initialize model
print("\n[3/5] Testing model initialization...")
try:
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
        d_model=128,  # Small for testing
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        positional_encoding='sinusoidal',
        max_seq_len=800
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {num_params:,} parameters")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass
print("\n[4/5] Testing forward pass...")
try:
    # Get a batch
    batch = next(iter(train_loader))
    temps = batch['temperatures']
    features = batch['features']
    mask = batch['mask']

    print(f"  Batch shape: {temps.shape}")
    print(f"  Mask shape: {mask.shape}")

    # Forward pass
    input_temps = temps[:, :-1]
    target_temps = temps[:, 1:].unsqueeze(-1)
    input_mask = mask[:, :-1]

    with torch.no_grad():
        predictions = model(input_temps, features, input_mask)

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Target shape: {target_temps.shape}")
    print("✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test generation
print("\n[5/5] Testing generation...")
try:
    # Get features from first batch
    batch = next(iter(val_loader))
    features = {
        'categorical': {k: v[:1] for k, v in batch['features']['categorical'].items()},
        'continuous': {k: v[:1] for k, v in batch['features']['continuous'].items()},
        'flavors': batch['features']['flavors'][:1]
    }

    # Generate short profile
    with torch.no_grad():
        generated = model.generate(
            features=features,
            start_temp=426.0,
            target_duration=100,  # Short for testing
            device='cpu'
        )

    print(f"  Generated profile: {len(generated)} timesteps")
    print(f"  Start temp: {generated[0]:.1f}°F")
    print(f"  Final temp: {generated[-1]:.1f}°F")
    print("✓ Generation successful")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)
print("\nYour RoastFormer pipeline is ready!")
print("\nNext steps:")
print("  1. Train the transformer: python train_transformer.py")
print("  2. Evaluate the model: python evaluate_transformer.py")
print("  3. Generate profiles: python generate_profiles.py --list_features")
print("="*80)
