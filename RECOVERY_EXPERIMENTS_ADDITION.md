# Recovery Experiments to Add to Training Suite

**Purpose**: Address model collapse issue discovered in evaluation

**Date**: November 18, 2025

---

## New Experiments to Add

Add these to the `EXPERIMENTS` configuration in Cell 12:

```python
EXPERIMENTS = {
    # ─── TIER 1: ORIGINAL (Already run, showed collapse) ───
    'baseline_sinusoidal': False,      # Already trained - showed collapse
    'learned_pe': False,                # Already trained
    'rope_pe': False,                   # Already trained
    'no_flavors': False,                # Already trained
    'small_model': False,               # d_model=128 - already trained

    # ─── TIER 2: RECOVERY EXPERIMENTS (New!) ───
    'tiny_model': True,                 # d_model=64, 3 layers (~400K params)
    'micro_model': True,                # d_model=32, 2 layers (~100K params)
    'low_lr': True,                     # Lower learning rate (1e-5)
    'high_dropout': True,               # More regularization (dropout=0.3)
    'combined_fix': True,               # Tiny model + low LR + high dropout
}
```

---

## Experiment Configurations to Add

### 1. Tiny Model (d_model=64)
```python
if EXPERIMENTS['tiny_model']:
    exp_config = BASE_CONFIG.copy()
    exp_config['d_model'] = 64  # Much smaller
    exp_config['num_layers'] = 3
    exp_config['nhead'] = 4
    exp_config['dim_feedforward'] = 256
    exp_config['positional_encoding'] = 'sinusoidal'
    exp_config['experiment_name'] = 'tiny_model_d64'
    exp_config['checkpoint_dir'] = 'checkpoints/recovery_tiny'
    exp_config['results_dir'] = 'results/recovery_tiny'
    experiment_configs['tiny_model'] = exp_config
```

**Rationale**: ~400K params for 123 samples = ~3,200 params/sample (healthy ratio)

### 2. Micro Model (d_model=32)
```python
if EXPERIMENTS['micro_model']:
    exp_config = BASE_CONFIG.copy()
    exp_config['d_model'] = 32  # Extremely small
    exp_config['num_layers'] = 2
    exp_config['nhead'] = 2
    exp_config['dim_feedforward'] = 128
    exp_config['positional_encoding'] = 'sinusoidal'
    exp_config['experiment_name'] = 'micro_model_d32'
    exp_config['checkpoint_dir'] = 'checkpoints/recovery_micro'
    exp_config['results_dir'] = 'results/recovery_micro'
    experiment_configs['micro_model'] = exp_config
```

**Rationale**: ~100K params for 123 samples = ~800 params/sample (very conservative)

### 3. Low Learning Rate
```python
if EXPERIMENTS['low_lr']:
    exp_config = BASE_CONFIG.copy()
    exp_config['d_model'] = 128  # Medium size
    exp_config['num_layers'] = 4
    exp_config['learning_rate'] = 1e-5  # 10x lower
    exp_config['positional_encoding'] = 'sinusoidal'
    exp_config['experiment_name'] = 'low_lr_1e5'
    exp_config['checkpoint_dir'] = 'checkpoints/recovery_low_lr'
    exp_config['results_dir'] = 'results/recovery_low_lr'
    experiment_configs['low_lr'] = exp_config
```

**Rationale**: Original 1e-4 may have caused early divergence

### 4. High Dropout
```python
if EXPERIMENTS['high_dropout']:
    exp_config = BASE_CONFIG.copy()
    exp_config['d_model'] = 128
    exp_config['num_layers'] = 4
    exp_config['dropout'] = 0.3  # 3x higher regularization
    exp_config['positional_encoding'] = 'sinusoidal'
    exp_config['experiment_name'] = 'high_dropout_03'
    exp_config['checkpoint_dir'] = 'checkpoints/recovery_dropout'
    exp_config['results_dir'] = 'results/recovery_dropout'
    experiment_configs['high_dropout'] = exp_config
```

**Rationale**: Prevent overfitting to spurious patterns

### 5. Combined Fix (Best Practices)
```python
if EXPERIMENTS['combined_fix']:
    exp_config = BASE_CONFIG.copy()
    exp_config['d_model'] = 64
    exp_config['num_layers'] = 3
    exp_config['nhead'] = 4
    exp_config['dim_feedforward'] = 256
    exp_config['learning_rate'] = 5e-5  # Conservative LR
    exp_config['dropout'] = 0.25  # High dropout
    exp_config['weight_decay'] = 0.05  # More L2 regularization
    exp_config['positional_encoding'] = 'sinusoidal'
    exp_config['experiment_name'] = 'combined_best_practices'
    exp_config['checkpoint_dir'] = 'checkpoints/recovery_combined'
    exp_config['results_dir'] = 'results/recovery_combined'
    experiment_configs['combined_fix'] = exp_config
```

**Rationale**: All fixes together for maximum stability

---

## Teacher Forcing Evaluation Cell

**Add this as a NEW cell after training completes:**

```python
# ═══════════════════════════════════════════════════════════════
# TEACHER FORCING EVALUATION
# ═══════════════════════════════════════════════════════════════
# Test if models work with real temperatures (not autoregressive)

print("="*80)
print("TEACHER FORCING EVALUATION")
print("="*80)
print("\nPurpose: Test if models learned patterns despite autoregressive failure")
print("Method: Feed models REAL previous temperatures, not their own predictions\n")

import torch
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader
from src.model.transformer_adapter import AdaptedConditioningModule, AdaptedRoastFormer
import numpy as np

# Load validation data
data_loader = PreprocessedDataLoader(preprocessed_dir='preprocessed_data')
train_profiles, val_profiles = data_loader.load_data()

val_loader = data_loader.create_dataloaders(batch_size=1, max_sequence_length=800)[1]

teacher_forcing_results = {}

for exp_name, exp_data in all_results.items():
    if exp_data['status'] != 'SUCCESS':
        continue

    print(f"\n{'─'*60}")
    print(f"Testing: {exp_name}")
    print(f"{'─'*60}")

    # Load checkpoint
    checkpoint_path = Path(exp_data['config']['checkpoint_dir']) / 'best_transformer_model.pt'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found")
        continue

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    config = checkpoint['config']
    feature_dims = data_loader.get_feature_dimensions()

    # Initialize model
    conditioning_module = AdaptedConditioningModule(
        num_origins=feature_dims['num_origins'],
        num_processes=feature_dims['num_processes'],
        num_roast_levels=feature_dims['num_roast_levels'],
        num_varieties=feature_dims['num_varieties'],
        num_flavors=feature_dims['num_flavors'],
        embed_dim=config['embed_dim']
    )

    model = AdaptedRoastFormer(
        conditioning_module=conditioning_module,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        positional_encoding=config['positional_encoding'],
        max_seq_len=config['max_sequence_length']
    ).to('cuda')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test with teacher forcing
    mae_scores = []

    with torch.no_grad():
        for batch in val_loader:
            if len(mae_scores) >= 10:  # Test 10 samples
                break

            temps = batch['temperatures'].to('cuda')
            mask = batch['mask'].to('cuda')

            features = {
                'categorical': {k: v.to('cuda') for k, v in batch['features']['categorical'].items()},
                'continuous': {k: v.to('cuda') for k, v in batch['features']['continuous'].items()},
                'flavors': batch['features']['flavors'].to('cuda')
            }

            # Teacher forcing: Use real temps as input
            input_temps = temps[:, :-1]
            target_temps = temps[:, 1:].unsqueeze(-1)
            input_mask = mask[:, :-1]

            # Forward pass
            predictions = model(input_temps, features, input_mask)

            # Compute MAE on valid positions
            loss_mask = input_mask.unsqueeze(-1).float()
            masked_predictions = predictions * loss_mask
            masked_targets = target_temps * loss_mask

            mae = torch.abs(masked_predictions - masked_targets).sum() / loss_mask.sum()
            mae_scores.append(mae.item())

    avg_mae = np.mean(mae_scores)
    teacher_forcing_results[exp_name] = {
        'mae': avg_mae,
        'd_model': config['d_model'],
        'params': sum(p.numel() for p in model.parameters())
    }

    print(f"Teacher Forcing MAE: {avg_mae:.2f}°F")

    # Interpretation
    if avg_mae < 10:
        print("✅ EXCELLENT - Model learned patterns very well!")
        print("   Issue is autoregressive compound error, not learning")
    elif avg_mae < 50:
        print("✅ GOOD - Model learned reasonable patterns")
        print("   Scheduled sampling could bridge the gap")
    elif avg_mae < 150:
        print("⚠️  MODERATE - Model learned some patterns")
        print("   May need architecture changes")
    else:
        print("❌ POOR - Model didn't learn effectively")
        print("   Fundamental training failure")

# Summary
print(f"\n{'='*80}")
print("TEACHER FORCING SUMMARY")
print(f"{'='*80}\n")

if teacher_forcing_results:
    tf_df = pd.DataFrame(teacher_forcing_results).T
    tf_df = tf_df.sort_values('mae')
    print(tf_df.to_string())

    best = tf_df.iloc[0]
    print(f"\n🏆 Best with Teacher Forcing: {tf_df.index[0]}")
    print(f"   MAE: {best['mae']:.2f}°F")
    print(f"   d_model: {int(best['d_model'])}")
    print(f"   Parameters: {int(best['params']):,}")

    if best['mae'] < 50:
        print(f"\n💡 CONCLUSION:")
        print(f"   Model CAN learn patterns but fails autoregressively.")
        print(f"   Fix: Implement scheduled sampling during training.")
    else:
        print(f"\n💡 CONCLUSION:")
        print(f"   Model struggles even with teacher forcing.")
        print(f"   Smaller models from recovery experiments should help.")

print("="*80)
```

---

## Diagnostic Cell: Generation Debug

**Add this cell to debug generation in detail:**

```python
# ═══════════════════════════════════════════════════════════════
# GENERATION DIAGNOSTIC - Compare All Models
# ═══════════════════════════════════════════════════════════════

print("="*80)
print("GENERATION DIAGNOSTIC - First 10 Steps")
print("="*80)

# Test generation for each model
for exp_name, exp_data in all_results.items():
    if exp_data['status'] != 'SUCCESS':
        continue

    print(f"\n{'─'*60}")
    print(f"{exp_name.upper()}")
    print(f"d_model={exp_data['config']['d_model']}, ")
    print(f"params={teacher_forcing_results.get(exp_name, {}).get('params', 'N/A')}")
    print(f"{'─'*60}")

    # Load model (code from teacher forcing eval above)
    # ... (same model loading code)

    # Get one validation sample
    batch = next(iter(val_loader))
    temps = batch['temperatures'].to('cuda')
    features = {...}  # Same as above

    start_temp = float(temps[0, 0])
    generated = torch.tensor([[start_temp]], device='cuda')

    print(f"Start: {start_temp:.1f}°F")

    with torch.no_grad():
        for t in range(10):
            output = model.forward(generated, features)
            next_temp_raw = output[0, -1, 0].item()
            next_temp_clamped = torch.clamp(output[0, -1, 0], min=100.0, max=500.0).item()

            print(f"  Step {t+1}: Raw={next_temp_raw:7.1f}°F, Clamped={next_temp_clamped:7.1f}°F")

            generated = torch.cat([generated, torch.tensor([[next_temp_clamped]], device='cuda')], dim=1)

    # Check if varying
    preds = [generated[0, i].item() for i in range(generated.shape[1])]
    variance = np.var(preds)
    print(f"\nVariance: {variance:.2f}")

    if variance < 1:
        print("❌ CONSTANT OUTPUT (collapsed)")
    elif variance < 100:
        print("⚠️  LOW VARIATION (weak generation)")
    else:
        print("✅ VARYING OUTPUT (good!)")

print("="*80)
```

---

## Expected Outcomes

### If Tiny/Micro Models Work:
- **Prediction**: Should avoid collapse, generate varying temps
- **MAE**: Expect 10-30°F with teacher forcing
- **Generation**: Should produce realistic curves
- **Conclusion**: Problem was model size vs data

### If Still Collapse:
- **Teacher forcing MAE high** (>100°F): Fundamental training issue
- **Teacher forcing MAE low** (<50°F): Autoregressive gap issue
- **Solution**: Implement scheduled sampling or try RNN architecture

---

## Integration Instructions

1. **Copy experiment configs** from sections 1-5 above
2. **Add to Cell 12** in existing training suite
3. **Add teacher forcing eval cell** after Cell 14
4. **Add generation diagnostic cell** after teacher forcing
5. **Run recovery experiments** with new configurations
6. **Compare results** to original experiments

---

## Timeline

**Tonight (Nov 18)**:
- Add recovery experiments to training suite
- Upload to Google Colab
- Run 2-3 promising configs (tiny, micro, combined)

**Tomorrow (Nov 19)**:
- Analyze recovery results
- Write findings in critical analysis
- Update evaluation framework

**If Recovery Succeeds**:
- Great! Use recovered model for evaluation
- Document what fixed it
- Present as "debugging deep learning" narrative

**If Recovery Fails**:
- Still valuable! Document systematic diagnosis
- Use teacher forcing results to show model CAN learn
- Discuss limitations and propose future fixes
- This is GOOD critical analysis!

---

**Bottom Line**: These recovery experiments give you multiple chances to get a working model, and even if they don't work, the systematic debugging shows deep understanding! 🎯
