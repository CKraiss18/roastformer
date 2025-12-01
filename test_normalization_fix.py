"""
Quick Test: Verify Normalization Fix Works

Trains micro model for 5 epochs to check if loss drops properly.

Expected with fix:
- Epoch 1: Loss ~0.5-1.0 (normalized MSE)
- Epoch 5: Loss ~0.1-0.3 (should drop by >50%)

Without fix (broken):
- Epoch 1: Loss ~78,000 (raw MSE)
- Epoch 5: Loss ~76,000 (only 2.5% drop)
"""

import sys
sys.path.append('.')

import torch
from train_transformer import TransformerTrainer

# Micro model config (fastest to test)
config = {
    'd_model': 32,
    'nhead': 2,
    'num_layers': 2,
    'dim_feedforward': 128,
    'embed_dim': 32,
    'dropout': 0.1,
    'batch_size': 8,
    'num_epochs': 5,  # Just 5 epochs to test
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'early_stopping_patience': 100,  # Don't stop early
    'max_sequence_length': 800,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'preprocessed_dir': 'preprocessed_data',
    'positional_encoding': 'sinusoidal',
    'experiment_name': 'test_normalization_fix',
    'checkpoint_dir': 'checkpoints/test_fix',
    'results_dir': 'results/test_fix',
    'save_every': 10
}

print("="*80)
print("TESTING NORMALIZATION FIX")
print("="*80)
print(f"\nDevice: {config['device']}")
print(f"Model: Micro (d={config['d_model']}, layers={config['num_layers']})")
print(f"Epochs: {config['num_epochs']}")
print("\n" + "="*80)
print("EXPECTED RESULTS WITH FIX:")
print("="*80)
print("  Epoch 1: Loss ~0.5-1.0 (normalized MSE)")
print("  Epoch 5: Loss ~0.1-0.3 (50%+ drop)")
print("  Generation: Varying temps (not constant)")
print("\n" + "="*80)
print("IF BROKEN (normalization not applied):")
print("="*80)
print("  Epoch 1: Loss ~78,000 (raw MSE)")
print("  Epoch 5: Loss ~76,000 (2.5% drop)")
print("  Generation: Constant ~5-10°F")
print("="*80 + "\n")

# Train
trainer = TransformerTrainer(config)
trainer.load_data()
trainer.initialize_model()
trainer.train()

# Check final loss
final_train_loss = trainer.train_losses[-1]
final_val_loss = trainer.val_losses[-1]

print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)
print(f"\nFinal Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Interpret
if final_val_loss < 5.0:
    print("\n✅ SUCCESS! Normalization is working!")
    print("   Loss is in normalized range [0, 1]")
    print("   Expected RMSE in real temps: ~" + f"{(final_val_loss**0.5) * 400:.1f}°F")
elif final_val_loss > 10000:
    print("\n❌ FAILED! Normalization NOT applied!")
    print("   Loss is in raw temperature range (broken)")
    print("   Check that imports are correct:")
    print("   - train_transformer.py imports preprocessed_data_loader_NORMALIZED")
    print("   - transformer_adapter.py generate() uses normalize/denormalize")
else:
    print("\n⚠️  UNCLEAR - Loss is between normalized and raw")
    print("   This shouldn't happen - check code")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

if final_val_loss < 5.0:
    print("\n✅ Fix verified! Now you can:")
    print("   1. Train tiny model (d=64) for better accuracy")
    print("   2. Test generation to ensure varying temps")
    print("   3. Run full evaluation")
else:
    print("\n❌ Fix not working. Debug:")
    print("   1. Check train_transformer.py line 21 imports NORMALIZED loader")
    print("   2. Check transformer_adapter.py generate() uses normalize/denormalize")
    print("   3. Try restarting Python kernel if in notebook")

print("="*80)
