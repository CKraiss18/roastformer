"""
Evaluation Script for Trained RoastFormer Transformer

Loads a trained transformer and evaluates it on:
- Validation set performance
- Physics-based constraints
- Generated profile quality
- Attention patterns (optional)

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Import data loader
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

# Import adapted transformer
from src.model.transformer_adapter import (
    AdaptedConditioningModule,
    AdaptedRoastFormer
)


class TransformerEvaluator:
    """
    Comprehensive evaluation for trained RoastFormer
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        print(f"\n📦 Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint['config']
        self.feature_dims = self.checkpoint['feature_dims']

        print(f"✓ Checkpoint loaded (Epoch {self.checkpoint['epoch']})")
        print(f"✓ Best val loss: {self.checkpoint['best_val_loss']:.4f}")

        # Initialize model
        self.model = self._initialize_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model initialized and loaded")

        # Data loaders (to be initialized)
        self.train_loader = None
        self.val_loader = None

    def _initialize_model(self) -> AdaptedRoastFormer:
        """Initialize model from checkpoint config"""
        conditioning_module = AdaptedConditioningModule(
            num_origins=self.feature_dims['num_origins'],
            num_processes=self.feature_dims['num_processes'],
            num_roast_levels=self.feature_dims['num_roast_levels'],
            num_varieties=self.feature_dims['num_varieties'],
            num_flavors=self.feature_dims['num_flavors'],
            embed_dim=self.config.get('embed_dim', 32)
        )

        model = AdaptedRoastFormer(
            conditioning_module=conditioning_module,
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 6),
            dim_feedforward=self.config.get('dim_feedforward', 1024),
            dropout=self.config.get('dropout', 0.1),
            positional_encoding=self.config.get('positional_encoding', 'sinusoidal'),
            max_seq_len=self.config.get('max_sequence_length', 1000)
        ).to(self.device)

        return model

    def load_data(self):
        """Load validation data"""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        data_loader = PreprocessedDataLoader(
            preprocessed_dir=self.config.get('preprocessed_dir', 'preprocessed_data')
        )

        # Load profiles
        train_profiles, val_profiles = data_loader.load_data()

        # Create dataloaders
        self.train_loader, self.val_loader = data_loader.create_dataloaders(
            batch_size=self.config['batch_size'],
            max_sequence_length=self.config.get('max_sequence_length', 1000)
        )

        print(f"✓ Validation set: {len(val_profiles)} profiles")

        return data_loader

    def evaluate_validation_set(self) -> Dict:
        """
        Evaluate model on validation set

        Returns:
            Dict with evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATING ON VALIDATION SET")
        print("="*80)

        criterion = nn.MSELoss()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        num_samples = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                temps = batch['temperatures'].to(self.device)
                mask = batch['mask'].to(self.device)

                features = {
                    'categorical': {k: v.to(self.device) for k, v in batch['features']['categorical'].items()},
                    'continuous': {k: v.to(self.device) for k, v in batch['features']['continuous'].items()},
                    'flavors': batch['features']['flavors'].to(self.device)
                }

                # Teacher forcing
                input_temps = temps[:, :-1]
                target_temps = temps[:, 1:].unsqueeze(-1)
                input_mask = mask[:, :-1]

                # Forward pass
                predictions = self.model(input_temps, features, input_mask)

                # Compute loss on valid positions
                loss_mask = input_mask.unsqueeze(-1).float()
                masked_predictions = predictions * loss_mask
                masked_targets = target_temps * loss_mask

                loss = criterion(masked_predictions, masked_targets)
                mae = torch.abs(masked_predictions - masked_targets).sum() / loss_mask.sum()

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
                num_samples += temps.shape[0]

                # Store for analysis
                all_predictions.append(predictions.cpu())
                all_targets.append(target_temps.cpu())

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        print(f"\n📊 Validation Metrics:")
        print(f"   MSE Loss: {avg_loss:.4f}")
        print(f"   MAE: {avg_mae:.4f}°F")
        print(f"   RMSE: {np.sqrt(avg_loss):.4f}°F")
        print(f"   Samples evaluated: {num_samples}")

        return {
            'mse_loss': avg_loss,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_loss),
            'num_samples': num_samples,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def generate_sample_profiles(self, num_samples: int = 5) -> List[Dict]:
        """
        Generate sample profiles from validation set

        Args:
            num_samples: Number of profiles to generate

        Returns:
            List of dicts with real and generated profiles
        """
        print("\n" + "="*80)
        print(f"GENERATING {num_samples} SAMPLE PROFILES")
        print("="*80)

        generated_profiles = []
        samples_generated = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if samples_generated >= num_samples:
                    break

                temps = batch['temperatures'].to(self.device)
                mask = batch['mask'].to(self.device)
                original_lengths = batch['original_length']
                product_names = batch['product_name']

                features = {
                    'categorical': {k: v.to(self.device) for k, v in batch['features']['categorical'].items()},
                    'continuous': {k: v.to(self.device) for k, v in batch['features']['continuous'].items()},
                    'flavors': batch['features']['flavors'].to(self.device)
                }

                # Generate for each sample in batch
                batch_size = temps.shape[0]
                for i in range(batch_size):
                    if samples_generated >= num_samples:
                        break

                    # Get single sample features
                    sample_features = {
                        'categorical': {k: v[i:i+1] for k, v in features['categorical'].items()},
                        'continuous': {k: v[i:i+1] for k, v in features['continuous'].items()},
                        'flavors': features['flavors'][i:i+1]
                    }

                    # Real profile
                    real_profile = temps[i, :original_lengths[i]].cpu().numpy()

                    # Generate profile
                    start_temp = real_profile[0]
                    target_duration = len(real_profile)

                    generated = self.model.generate(
                        features=sample_features,
                        start_temp=float(start_temp),
                        target_duration=target_duration,
                        device=self.device
                    )

                    # Validate physics constraints
                    physics_check = self._check_physics_constraints(generated)

                    generated_profiles.append({
                        'product_name': product_names[i],
                        'real_profile': real_profile,
                        'generated_profile': generated,
                        'physics_check': physics_check
                    })

                    print(f"\n✓ Generated profile {samples_generated + 1}: {product_names[i]}")
                    print(f"   Length: {len(generated)} steps")
                    print(f"   Start temp: {generated[0]:.1f}°F")
                    print(f"   Final temp: {generated[-1]:.1f}°F")
                    print(f"   Physics valid: {physics_check['all_valid']}")

                    samples_generated += 1

        return generated_profiles

    def _check_physics_constraints(self, temps: np.ndarray) -> Dict:
        """
        Check if generated profile satisfies physics constraints

        Args:
            temps: Temperature sequence

        Returns:
            Dict with constraint checks
        """
        checks = {}

        # 1. Temperature range
        checks['temp_range_valid'] = (temps >= 250).all() and (temps <= 450).all()
        checks['min_temp'] = float(temps.min())
        checks['max_temp'] = float(temps.max())

        # 2. Monotonicity (post-turning point)
        turning_idx = np.argmin(temps)
        post_turning = temps[turning_idx:]
        checks['monotonic_post_turning'] = (np.diff(post_turning) >= 0).all()
        checks['turning_point_idx'] = int(turning_idx)

        # 3. Heating rates (Rate of Rise)
        ror = np.diff(temps) * 60  # Convert to °F/min
        checks['ror_bounded'] = ((ror >= 20) & (ror <= 100)).mean() > 0.95
        checks['ror_mean'] = float(ror.mean())
        checks['ror_std'] = float(ror.std())

        # 4. No sudden jumps
        checks['no_sudden_jumps'] = (np.abs(np.diff(temps)) < 10/60).all()
        checks['max_jump'] = float(np.abs(np.diff(temps)).max())

        # 5. Overall validity
        checks['all_valid'] = all([
            checks['temp_range_valid'],
            checks['monotonic_post_turning'],
            checks['ror_bounded'],
            checks['no_sudden_jumps']
        ])

        return checks

    def save_evaluation_results(self, results: Dict, output_dir: str = 'results/evaluation'):
        """Save evaluation results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'evaluation_{timestamp}.json'

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # Handle generated_profiles
                results_serializable[key] = []
                for item in value:
                    item_copy = {}
                    for k, v in item.items():
                        if isinstance(v, np.ndarray):
                            item_copy[k] = v.tolist()
                        else:
                            item_copy[k] = v
                    results_serializable[key].append(item_copy)
            else:
                results_serializable[key] = value

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n✓ Evaluation results saved to {output_file}")

    def plot_sample_profiles(self, generated_profiles: List[Dict], output_dir: str = 'results/evaluation'):
        """Create comparison plots for generated profiles"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, profile_data in enumerate(generated_profiles):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            real = profile_data['real_profile']
            generated = profile_data['generated_profile']

            # Plot 1: Temperature profiles
            ax1.plot(real, label='Real Profile', linewidth=2, alpha=0.8)
            ax1.plot(generated, label='Generated Profile', linewidth=2, alpha=0.8, linestyle='--')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Temperature (°F)')
            ax1.set_title(f"{profile_data['product_name']}\nReal vs Generated")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Rate of Rise comparison
            real_ror = np.diff(real) * 60
            gen_ror = np.diff(generated) * 60

            ax2.plot(real_ror, label='Real RoR', linewidth=2, alpha=0.8)
            ax2.plot(gen_ror, label='Generated RoR', linewidth=2, alpha=0.8, linestyle='--')
            ax2.axhline(y=20, color='red', linestyle=':', alpha=0.5, label='Min RoR (20°F/min)')
            ax2.axhline(y=100, color='red', linestyle=':', alpha=0.5, label='Max RoR (100°F/min)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Rate of Rise (°F/min)')
            ax2.set_title('Rate of Rise Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'profile_comparison_{i+1}.png', dpi=150)
            plt.close()

        print(f"✓ Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RoastFormer')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_transformer_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample profiles to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')

    args = parser.parse_args()

    print("="*80)
    print("ROASTFORMER TRANSFORMER EVALUATION")
    print("="*80)

    # Initialize evaluator
    evaluator = TransformerEvaluator(args.checkpoint, args.device)

    # Load data
    evaluator.load_data()

    # Evaluate on validation set
    val_results = evaluator.evaluate_validation_set()

    # Generate sample profiles
    generated_profiles = evaluator.generate_sample_profiles(args.num_samples)

    # Compile results
    results = {
        'checkpoint': str(args.checkpoint),
        'validation_metrics': val_results,
        'generated_profiles': generated_profiles,
        'num_samples_generated': len(generated_profiles)
    }

    # Save results
    evaluator.save_evaluation_results(results, args.output_dir)

    # Create plots if requested
    if args.plot:
        evaluator.plot_sample_profiles(generated_profiles, args.output_dir)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Validation MAE: {val_results['mae']:.4f}°F")
    print(f"Validation RMSE: {val_results['rmse']:.4f}°F")
    print(f"Sample profiles generated: {len(generated_profiles)}")

    physics_valid = sum(1 for p in generated_profiles if p['physics_check']['all_valid'])
    print(f"Physics-valid profiles: {physics_valid}/{len(generated_profiles)}")

    print("="*80)


if __name__ == "__main__":
    main()
