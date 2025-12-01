"""
Full Transformer Training Script for RoastFormer

Trains the complete transformer architecture (not the MLP baseline).
Uses the AdaptedRoastFormer with PreprocessedDataLoader.

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime
import argparse
from typing import Dict

# Import data loader
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

# Import adapted transformer
from src.model.transformer_adapter import (
    AdaptedConditioningModule,
    AdaptedRoastFormer
)


class TransformerTrainer:
    """
    Training pipeline for full RoastFormer transformer
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])

        # Data loaders (to be initialized)
        self.train_loader = None
        self.val_loader = None

        # Model (to be initialized)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load preprocessed data and create dataloaders"""
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

        # Get feature dimensions
        self.feature_dims = data_loader.get_feature_dimensions()

        print(f"\n📊 Dataset Statistics:")
        print(f"   Training samples: {len(train_profiles)}")
        print(f"   Validation samples: {len(val_profiles)}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")

        print(f"\n📏 Feature Dimensions:")
        for key, value in self.feature_dims.items():
            print(f"   {key}: {value}")

        return data_loader

    def initialize_model(self):
        """Initialize the full transformer model"""
        print("\n" + "="*80)
        print("INITIALIZING TRANSFORMER MODEL")
        print("="*80)

        # Create conditioning module
        conditioning_module = AdaptedConditioningModule(
            num_origins=self.feature_dims['num_origins'],
            num_processes=self.feature_dims['num_processes'],
            num_roast_levels=self.feature_dims['num_roast_levels'],
            num_varieties=self.feature_dims['num_varieties'],
            num_flavors=self.feature_dims['num_flavors'],
            embed_dim=self.config.get('embed_dim', 32)
        )

        # Create full transformer
        self.model = AdaptedRoastFormer(
            conditioning_module=conditioning_module,
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 6),
            dim_feedforward=self.config.get('dim_feedforward', 1024),
            dropout=self.config.get('dropout', 0.1),
            positional_encoding=self.config.get('positional_encoding', 'sinusoidal'),
            max_seq_len=self.config.get('max_sequence_length', 1000)
        ).to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n✓ Model Architecture:")
        print(f"   Type: Decoder-only Transformer")
        print(f"   d_model: {self.config.get('d_model', 256)}")
        print(f"   Attention heads: {self.config.get('nhead', 8)}")
        print(f"   Layers: {self.config.get('num_layers', 6)}")
        print(f"   Feedforward dim: {self.config.get('dim_feedforward', 1024)}")
        print(f"   Positional encoding: {self.config.get('positional_encoding', 'sinusoidal')}")
        print(f"   Dropout: {self.config.get('dropout', 0.1)}")

        print(f"\n✓ Model Parameters:")
        print(f"   Total: {num_params:,}")
        print(f"   Trainable: {num_trainable:,}")

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=self.config.get('betas', (0.9, 0.999))
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config.get('min_lr', 1e-6)
        )

        print(f"\n✓ Optimizer: AdamW")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Weight decay: {self.config.get('weight_decay', 0.01)}")
        print(f"   Scheduler: CosineAnnealingLR")

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move data to device
            temps = batch['temperatures'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Move features to device
            features = {
                'categorical': {k: v.to(self.device) for k, v in batch['features']['categorical'].items()},
                'continuous': {k: v.to(self.device) for k, v in batch['features']['continuous'].items()},
                'flavors': batch['features']['flavors'].to(self.device)
            }

            # Teacher forcing: predict next temperature from current
            # Input: temps[:-1], Target: temps[1:]
            input_temps = temps[:, :-1]
            target_temps = temps[:, 1:].unsqueeze(-1)
            input_mask = mask[:, :-1]

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(input_temps, features, input_mask)

            # Compute loss only on valid (unmasked) positions
            loss_mask = input_mask.unsqueeze(-1).float()
            masked_predictions = predictions * loss_mask
            masked_targets = target_temps * loss_mask

            loss = self.criterion(masked_predictions, masked_targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

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

                # Compute loss
                loss_mask = input_mask.unsqueeze(-1).float()
                masked_predictions = predictions * loss_mask
                masked_targets = target_temps * loss_mask

                loss = self.criterion(masked_predictions, masked_targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)

        for epoch in range(self.config['num_epochs']):
            epoch_start = datetime.now()
            self.current_epoch = epoch + 1

            print(f"\nEpoch {self.current_epoch}/{self.config['num_epochs']}")
            print("-" * 80)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step()

            # Print summary
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Time:       {epoch_time:.1f}s")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best model! (val_loss: {self.best_val_loss:.4f})")

            # Save regular checkpoint every N epochs
            if self.current_epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(is_best=False)

            # Early stopping check
            if self.config.get('early_stopping_patience'):
                if len(self.val_losses) > self.config['early_stopping_patience']:
                    recent_losses = self.val_losses[-self.config['early_stopping_patience']:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"\n⚠️  Early stopping triggered at epoch {self.current_epoch}")
                        break

        # Training complete
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        print(f"Final val loss: {self.val_losses[-1]:.4f}")

        # Save final results
        self.save_results()

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'feature_dims': self.feature_dims
        }

        if is_best:
            path = self.checkpoint_dir / "best_transformer_model.pt"
            torch.save(checkpoint, path)
        else:
            path = self.checkpoint_dir / f"transformer_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, path)

    def save_results(self):
        """Save training results"""
        results = {
            'config': self.config,
            'feature_dims': self.feature_dims,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }

        with open(self.results_dir / "transformer_training_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {self.results_dir / 'transformer_training_results.json'}")
        print(f"✓ Best model saved to {self.checkpoint_dir / 'best_transformer_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description='Train RoastFormer Transformer')

    # Data parameters
    parser.add_argument('--preprocessed_dir', type=str, default='preprocessed_data',
                        help='Directory with preprocessed data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--max_sequence_length', type=int, default=800,
                        help='Maximum sequence length')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Feedforward dimension')
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='Embedding dimension for categorical features')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--positional_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'learned'],
                        help='Type of positional encoding')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience (None to disable)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    config = vars(args)

    print("="*80)
    print("ROASTFORMER TRANSFORMER TRAINING")
    print("="*80)

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Initialize trainer
    trainer = TransformerTrainer(config)

    # Load data
    trainer.load_data()

    # Initialize model
    trainer.initialize_model()

    # Train
    trainer.train()

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Evaluate model: python evaluate_transformer.py")
    print("2. Generate profiles: python generate_profiles.py")
    print("3. Run ablation studies: python train_transformer.py --positional_encoding learned")
    print("="*80)


if __name__ == "__main__":
    main()
