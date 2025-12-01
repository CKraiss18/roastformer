"""
Train Tiny Model (d=64) with Normalization Fix

Better accuracy than micro model (d=32).
Expected: RMSE ~30-50°F, smooth generation curves.
"""

import sys
sys.path.append('.')

import torch
from train_transformer import TransformerTrainer

# Tiny model config (balanced: speed vs accuracy)
config = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 256,
    'embed_dim': 32,
    'dropout': 0.2,  # Higher dropout for better regularization
    'batch_size': 8,
    'num_epochs': 20,  # Train longer for convergence
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'early_stopping_patience': 15,
    'max_sequence_length': 800,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'preprocessed_dir': 'preprocessed_data',
    'positional_encoding': 'sinusoidal',
    'experiment_name': 'tiny_model_normalized',
    'checkpoint_dir': 'checkpoints/tiny_normalized',
    'results_dir': 'results/tiny_normalized',
    'save_every': 5
}

print("="*80)
print("TRAINING TINY MODEL (d=64) WITH NORMALIZATION")
print("="*80)
print(f"\nDevice: {config['device']}")
print(f"Model: Tiny (d={config['d_model']}, layers={config['num_layers']})")
print(f"Parameters: ~218K (1,775 params/sample - healthy!)")
print(f"Epochs: {config['num_epochs']}")
print(f"Dropout: {config['dropout']} (higher regularization)")
print("\n" + "="*80)
print("EXPECTED RESULTS:")
print("="*80)
print("  Epoch 10: RMSE ~0.15 normalized (~60°F real)")
print("  Epoch 20: RMSE ~0.08 normalized (~30°F real)")
print("  Generation: Smooth, varying curves")
print("  Teacher forcing MAE: <20°F")
print("="*80 + "\n")

# Train
trainer = TransformerTrainer(config)
trainer.load_data()
trainer.initialize_model()
trainer.train()

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE - SUMMARY")
print("="*80)

final_train_loss = trainer.train_losses[-1]
final_val_loss = trainer.val_losses[-1]
best_val_loss = trainer.best_val_loss

print(f"\nFinal Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")

# Convert to real RMSE
rmse_norm = best_val_loss ** 0.5
rmse_real = rmse_norm * 400  # Denormalize
print(f"\nBest RMSE (normalized): {rmse_norm:.4f}")
print(f"Best RMSE (real temps): {rmse_real:.1f}°F")

# Interpret
if rmse_real < 40:
    print("\n✅✅ EXCELLENT! RMSE <40°F")
    print("   This is production-quality accuracy")
    print("   Generation should be very smooth")
elif rmse_real < 60:
    print("\n✅ GOOD! RMSE 40-60°F")
    print("   Solid performance, usable for roasting guidance")
    print("   Generation should be smooth with minor variations")
elif rmse_real < 100:
    print("\n⚠️  ACCEPTABLE. RMSE 60-100°F")
    print("   Model learned patterns but needs improvement")
    print("   Generation may have some noise")
else:
    print("\n⚠️  NEEDS WORK. RMSE >100°F")
    print("   Model is learning but accuracy is low")
    print("   May need more training or larger model")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

if rmse_real < 60:
    print("\n✅ Model ready for evaluation!")
    print("   1. Test generation: python test_generation_tiny.py")
    print("   2. Run full evaluation with this model")
    print("   3. Create visualizations for presentation")
else:
    print("\n⚠️  Consider:")
    print("   1. Train for more epochs (if stopped early)")
    print("   2. Try larger model (d=128) if time permits")
    print("   3. Still usable for demonstration of fix!")

print("="*80)
