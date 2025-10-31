"""
Training Pipeline for RoastFormer

Complete training loop with:
- Data loading and preprocessing
- Model initialization
- Training with validation
- Checkpointing and logging
- Metrics tracking

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.data_preparation import RoastProfileDataLoader
from src.model.roastformer import (
    RoastFormer, ConditioningModule, FeatureEncoder,
    RoastProfileDataset, PositionalEncoding
)
from src.utils.validation import RoastProfileValidator
from src.utils.metrics import comprehensive_evaluation, evaluate_batch
from src.utils.visualization import plot_training_curves, plot_comparison

class RoastFormerTrainer:
    """
    Complete training pipeline for RoastFormer
    """

    def __init__(
        self,
        model: RoastFormer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results"
    ):
        """
        Initialize trainer

        Args:
            model: RoastFormer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            checkpoint_dir: Directory to save checkpoints
            results_dir: Directory to save results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            'mae': [],
            'finish_temp_error': []
        }

        print(f"✓ Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (temps, conditions, targets) in enumerate(self.train_loader):
            # Move to device
            temps = temps.to(self.device)
            conditions = conditions.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(temps, conditions)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self) -> Tuple[float, Dict]:
        """
        Validate for one epoch

        Returns:
            (avg_val_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_real = []
        all_generated = []

        with torch.no_grad():
            for temps, conditions, targets in self.val_loader:
                # Move to device
                temps = temps.to(self.device)
                conditions = conditions.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(temps, conditions)

                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

                # Collect for metrics (convert to numpy)
                all_real.extend([t.cpu().numpy() for t in targets])
                all_generated.extend([o.cpu().numpy() for o in outputs])

        avg_loss = total_loss / num_batches

        # Compute detailed metrics on a subset (for speed)
        sample_size = min(10, len(all_real))
        sample_real = all_real[:sample_size]
        sample_gen = all_generated[:sample_size]

        # Flatten sequences for metrics
        sample_real_flat = [seq.flatten() for seq in sample_real]
        sample_gen_flat = [seq.flatten() for seq in sample_gen]

        batch_metrics = evaluate_batch(sample_real_flat, sample_gen_flat, verbose=False)

        return avg_loss, batch_metrics['aggregate']

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            is_best: If True, save as best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")

        # Save latest
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.metrics_history = checkpoint['metrics_history']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, num_epochs: Optional[int] = None):
        """
        Complete training loop

        Args:
            num_epochs: Number of epochs (overrides config if provided)
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']

        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("=" * 80 + "\n")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = datetime.now()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, metrics = self.validate_epoch()
            self.val_losses.append(val_loss)

            # Track metrics
            self.metrics_history['mae'].append(metrics['mae']['mean'])
            self.metrics_history['finish_temp_error'].append(metrics['finish_temp_error']['mean'])

            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch summary
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MAE:        {metrics['mae']['mean']:.2f}°F")
            print(f"  Finish Err: {metrics['finish_temp_error']['mean']:.2f}°F")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.config.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.config.get('early_stopping', None):
                patience = self.config['early_stopping']
                if len(self.val_losses) > patience:
                    recent_losses = self.val_losses[-patience:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"\n⚠️  Early stopping triggered (patience: {patience})")
                        break

        # Training complete
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        print(f"Final val loss: {self.val_losses[-1]:.4f}")

        # Save final results
        self.save_results()

    def save_results(self):
        """Save training results and plots"""
        # Save training curves
        plot_path = self.results_dir / "training_curves.png"
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            metrics={'MAE': self.metrics_history['mae']},
            save_path=str(plot_path)
        )

        # Save metrics
        results = {
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch
        }

        results_path = self.results_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {self.results_dir}/")


def prepare_data(dataset_dirs: List[str], config: Dict) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Load and prepare data for training

    Args:
        dataset_dirs: List of dataset directories
        config: Training configuration

    Returns:
        (train_loader, val_loader, dataset_info)
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA")
    print("=" * 80)

    # Load data
    loader = RoastProfileDataLoader(auto_discover=False)

    all_profiles = []
    all_metadata = []

    for dataset_dir in dataset_dirs:
        profiles, metadata = loader.load_dataset(dataset_dir)
        all_profiles.extend(profiles)
        all_metadata.append(metadata)
        print(f"✓ Loaded {len(profiles)} profiles from {dataset_dir}")

    print(f"\n✓ Total profiles: {len(all_profiles)}")

    # Validate profiles
    validator = RoastProfileValidator(strict=False)
    validation_summary = validator.validate_dataset(all_profiles, verbose=True)

    if validation_summary['invalid'] > 0:
        print(f"\n⚠️  Warning: {validation_summary['invalid']} invalid profiles detected")
        # Filter out invalid profiles
        valid_indices = [i for i, r in enumerate(validation_summary['validation_results']) if r['valid']]
        all_profiles = [all_profiles[i] for i in valid_indices]
        print(f"✓ Filtered to {len(all_profiles)} valid profiles")

    # Create dataset
    # (This is simplified - you'll need to properly prepare features)
    # For now, returning mock data to show structure

    dataset_info = {
        'num_profiles': len(all_profiles),
        'num_valid': len(all_profiles),
        'num_invalid': validation_summary['invalid']
    }

    # Split into train/val
    val_size = int(len(all_profiles) * config['val_split'])
    train_size = len(all_profiles) - val_size

    print(f"\n✓ Train/val split: {train_size}/{val_size}")

    # TODO: Create actual PyTorch datasets and loaders
    # This is a placeholder - you'll need to integrate with actual data preparation

    return None, None, dataset_info


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train RoastFormer")
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='Dataset directories to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    args = parser.parse_args()

    # Configuration
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'val_split': 0.2,
        'grad_clip': 1.0,
        'weight_decay': 0.01,
        'save_every': 10,
        'early_stopping': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("\n" + "=" * 80)
    print("ROASTFORMER TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Prepare data
    train_loader, val_loader, dataset_info = prepare_data(args.datasets, config)

    # TODO: Initialize model properly with actual feature counts
    # This is a placeholder
    print("\n⚠️  Training pipeline structure is ready!")
    print("⚠️  Next step: Integrate actual data preparation and model initialization")
    print("⚠️  See src/dataset/data_preparation.py and src/model/roastformer.py")


if __name__ == "__main__":
    main()
