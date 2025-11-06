"""
Baseline Training Script for RoastFormer

Simple training script that uses preprocessed data and trains a basic model.
This is for testing the full pipeline end-to-end.

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime

# Import data loader
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

# For now, let's create a simple MLP baseline instead of full transformer
# to test the pipeline quickly
class SimpleRoastModel(nn.Module):
    """
    Simple MLP baseline for testing the training pipeline

    Takes conditioning features and predicts next temperature
    """

    def __init__(
        self,
        num_origins: int,
        num_processes: int,
        num_roast_levels: int,
        num_varieties: int,
        num_flavors: int,
        embed_dim: int = 32,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Categorical embeddings
        self.origin_embed = nn.Embedding(num_origins, embed_dim)
        self.process_embed = nn.Embedding(num_processes, embed_dim)
        self.roast_level_embed = nn.Embedding(num_roast_levels, embed_dim)
        self.variety_embed = nn.Embedding(num_varieties, embed_dim)

        # Calculate total conditioning dimension
        # 4 categorical * embed_dim + 3 continuous + num_flavors
        condition_dim = (4 * embed_dim) + 3 + num_flavors

        # Simple MLP for auto-regressive prediction
        # Input: current_temp + conditioning
        # Output: next_temp
        self.model = nn.Sequential(
            nn.Linear(1 + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, temps, features):
        """
        Forward pass

        Args:
            temps: Current temperatures (batch_size, seq_len)
            features: Dict of conditioning features

        Returns:
            predictions: (batch_size, seq_len, 1)
        """
        batch_size, seq_len = temps.shape

        # Get categorical embeddings
        origin_emb = self.origin_embed(features['categorical']['origin'].squeeze(-1))  # (batch, embed)
        process_emb = self.process_embed(features['categorical']['process'].squeeze(-1))
        roast_level_emb = self.roast_level_embed(features['categorical']['roast_level'].squeeze(-1))
        variety_emb = self.variety_embed(features['categorical']['variety'].squeeze(-1))

        # Concatenate all conditioning
        condition = torch.cat([
            origin_emb,
            process_emb,
            roast_level_emb,
            variety_emb,
            features['continuous']['target_finish_temp'],
            features['continuous']['altitude'],
            features['continuous']['bean_density'],
            features['flavors']
        ], dim=1)  # (batch, condition_dim)

        # Expand condition for all timesteps
        condition_expanded = condition.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, condition_dim)

        # Add temperature as input
        temps_input = temps.unsqueeze(-1)  # (batch, seq_len, 1)

        # Concatenate
        model_input = torch.cat([temps_input, condition_expanded], dim=2)  # (batch, seq_len, 1 + condition_dim)

        # Predict
        predictions = self.model(model_input)  # (batch, seq_len, 1)

        return predictions


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        temps = batch['temperatures'].to(device)
        mask = batch['mask'].to(device)

        # Move features to device
        features = {
            'categorical': {k: v.to(device) for k, v in batch['features']['categorical'].items()},
            'continuous': {k: v.to(device) for k, v in batch['features']['continuous'].items()},
            'flavors': batch['features']['flavors'].to(device)
        }

        # Create targets: shift by 1 (predict next temperature)
        # Input: temps[:-1], Target: temps[1:]
        input_temps = temps[:, :-1]
        target_temps = temps[:, 1:].unsqueeze(-1)
        input_mask = mask[:, :-1]

        # Forward pass
        optimizer.zero_grad()
        predictions = model(input_temps, features)

        # Compute loss only on valid (unmasked) positions
        loss_mask = input_mask.unsqueeze(-1).float()
        loss = criterion(predictions * loss_mask, target_temps * loss_mask)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            temps = batch['temperatures'].to(device)
            mask = batch['mask'].to(device)

            features = {
                'categorical': {k: v.to(device) for k, v in batch['features']['categorical'].items()},
                'continuous': {k: v.to(device) for k, v in batch['features']['continuous'].items()},
                'flavors': batch['features']['flavors'].to(device)
            }

            # Create targets
            input_temps = temps[:, :-1]
            target_temps = temps[:, 1:].unsqueeze(-1)
            input_mask = mask[:, :-1]

            # Forward pass
            predictions = model(input_temps, features)

            # Compute loss
            loss_mask = input_mask.unsqueeze(-1).float()
            loss = criterion(predictions * loss_mask, target_temps * loss_mask)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    print("\n" + "="*80)
    print("ROASTFORMER BASELINE TRAINING")
    print("="*80)

    # Configuration
    config = {
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'embed_dim': 32,
        'hidden_dim': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")

    # Load preprocessed data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    data_loader = PreprocessedDataLoader(preprocessed_dir="preprocessed_data")
    train_profiles, val_profiles = data_loader.load_data()

    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders(
        batch_size=config['batch_size'],
        max_sequence_length=1000
    )

    # Get feature dimensions
    dims = data_loader.get_feature_dimensions()
    print(f"\nFeature dimensions:")
    for key, value in dims.items():
        print(f"  {key}: {value}")

    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)

    model = SimpleRoastModel(
        num_origins=dims['num_origins'],
        num_processes=dims['num_processes'],
        num_roast_levels=dims['num_roast_levels'],
        num_varieties=dims['num_varieties'],
        num_flavors=dims['num_flavors'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  Parameters: {num_params:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    criterion = nn.MSELoss()

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(config['num_epochs']):
        epoch_start = datetime.now()

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Print summary
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time:       {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': config,
                'feature_dims': dims
            }
            torch.save(checkpoint, checkpoint_dir / "best_baseline_model.pt")
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")

    # Training complete
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")

    # Save results
    results = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'feature_dims': dims
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "baseline_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to results/baseline_training_results.json")
    print(f"✓ Best model saved to checkpoints/best_baseline_model.pt")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review training curves in results/")
    print("2. Test generation from trained model")
    print("3. Continue collecting more data daily")
    print("4. Retrain with larger dataset for better performance")
    print("="*80)


if __name__ == "__main__":
    main()
