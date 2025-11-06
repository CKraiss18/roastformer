"""
Preprocessed Data Loader for RoastFormer Training

Loads the preprocessed train/val profiles from prepare_training_data.py output.
Handles feature encoding and creates PyTorch datasets ready for training.

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class PreprocessedRoastDataset(Dataset):
    """
    PyTorch Dataset for preprocessed roast profiles

    Loads from train_profiles.json or val_profiles.json
    """

    def __init__(
        self,
        profiles: List[Dict],
        metadata_df: pd.DataFrame,
        encoders: Dict,
        flavor_vocab: Dict,
        max_sequence_length: int = 1000
    ):
        """
        Args:
            profiles: List of profile dicts from JSON
            metadata_df: DataFrame with metadata
            encoders: Dict of LabelEncoders for categorical features
            flavor_vocab: Dict mapping flavor names to indices
            max_sequence_length: Max length for temperature sequences
        """
        self.profiles = profiles
        self.metadata_df = metadata_df
        self.encoders = encoders
        self.flavor_vocab = flavor_vocab
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        """
        Get a single training example

        Returns:
            temps: Temperature sequence (padded/truncated to max_length)
            features: Dict of encoded features
            mask: Boolean mask for valid timesteps
        """
        profile = self.profiles[idx]

        # Extract temperature sequence
        bean_temp_data = profile['roast_profile']['bean_temp']
        temps = np.array([point['value'] for point in bean_temp_data])

        # Store original length for masking
        original_length = len(temps)

        # Truncate or pad
        if len(temps) > self.max_sequence_length:
            temps = temps[:self.max_sequence_length]
            original_length = self.max_sequence_length
        elif len(temps) < self.max_sequence_length:
            # Pad with last temperature (common for sequences)
            padding = np.full(self.max_sequence_length - len(temps), temps[-1])
            temps = np.concatenate([temps, padding])

        # Create mask (True for valid timesteps)
        mask = np.zeros(self.max_sequence_length, dtype=bool)
        mask[:original_length] = True

        # Extract metadata
        metadata = profile['metadata']
        product_name = metadata.get('product_name', 'Unknown')

        # Find matching row in metadata_df
        matching_rows = self.metadata_df[
            self.metadata_df['product_name'] == product_name
        ]

        if matching_rows.empty:
            # Use defaults if no match
            row = self._get_default_features()
        else:
            row = matching_rows.iloc[0]

        # Encode features
        features = self._encode_features(row, metadata)

        # Convert to tensors
        temps_tensor = torch.FloatTensor(temps)
        mask_tensor = torch.BoolTensor(mask)

        return {
            'temperatures': temps_tensor,
            'features': features,
            'mask': mask_tensor,
            'original_length': original_length,
            'product_name': product_name
        }

    def _encode_features(self, row: pd.Series, metadata: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode categorical and continuous features

        Returns:
            Dict of feature tensors
        """
        # Categorical features (encoded as indices)
        categorical_features = {}

        # Origin
        origin = row.get('origin', metadata.get('origin', 'Unknown'))
        if origin in self.encoders['origin'].classes_:
            categorical_features['origin'] = torch.LongTensor([
                self.encoders['origin'].transform([origin])[0]
            ])
        else:
            categorical_features['origin'] = torch.LongTensor([0])

        # Process
        process = row.get('process', metadata.get('process', 'Unknown'))
        if process in self.encoders['process'].classes_:
            categorical_features['process'] = torch.LongTensor([
                self.encoders['process'].transform([process])[0]
            ])
        else:
            categorical_features['process'] = torch.LongTensor([0])

        # Roast Level
        roast_level = row.get('roast_level', metadata.get('roast_level', 'Unknown'))
        if roast_level in self.encoders['roast_level'].classes_:
            categorical_features['roast_level'] = torch.LongTensor([
                self.encoders['roast_level'].transform([roast_level])[0]
            ])
        else:
            categorical_features['roast_level'] = torch.LongTensor([0])

        # Variety
        variety = row.get('variety', metadata.get('variety', 'Unknown'))
        if variety in self.encoders['variety'].classes_:
            categorical_features['variety'] = torch.LongTensor([
                self.encoders['variety'].transform([variety])[0]
            ])
        else:
            categorical_features['variety'] = torch.LongTensor([0])

        # Continuous features (normalized)
        continuous_features = {}

        # Target finish temp (normalize to ~0-1 range)
        target_temp = row.get('target_finish_temp', metadata.get('target_finish_temp', 395))
        try:
            target_temp = float(target_temp) if target_temp else 395.0
        except (ValueError, TypeError):
            target_temp = 395.0
        continuous_features['target_finish_temp'] = torch.FloatTensor([target_temp / 425.0])

        # Altitude (normalize, using default 1500 MASL if missing)
        altitude = metadata.get('altitude', 1500)
        if pd.isna(altitude) or altitude == '' or altitude is None:
            altitude = 1500
        try:
            altitude = float(altitude)
        except (ValueError, TypeError):
            altitude = 1500
        continuous_features['altitude'] = torch.FloatTensor([altitude / 2500.0])

        # Bean density proxy (default 0.68)
        bean_density = metadata.get('bean_density_proxy', 0.68)
        if pd.isna(bean_density) or bean_density == '' or bean_density is None:
            bean_density = 0.68
        try:
            bean_density = float(bean_density)
        except (ValueError, TypeError):
            bean_density = 0.68
        continuous_features['bean_density'] = torch.FloatTensor([bean_density / 0.80])

        # Flavor features (multi-hot encoding)
        flavor_vector = self._encode_flavors(metadata.get('flavor_notes_parsed', []))

        return {
            'categorical': categorical_features,
            'continuous': continuous_features,
            'flavors': flavor_vector
        }

    def _encode_flavors(self, flavor_notes_parsed) -> torch.Tensor:
        """
        Create multi-hot flavor vector

        Args:
            flavor_notes_parsed: List of flavor strings or comma-separated string

        Returns:
            Multi-hot tensor of size (len(flavor_vocab),)
        """
        flavor_vector = torch.zeros(len(self.flavor_vocab))

        # Handle both list and string formats
        if isinstance(flavor_notes_parsed, list):
            flavors = [f.strip().lower() for f in flavor_notes_parsed]
        elif isinstance(flavor_notes_parsed, str):
            flavors = [f.strip().lower() for f in flavor_notes_parsed.split(',')]
        else:
            flavors = []

        # Set indices to 1 for present flavors
        for flavor in flavors:
            if flavor in self.flavor_vocab:
                idx = self.flavor_vocab[flavor]
                flavor_vector[idx] = 1.0

        return flavor_vector

    def _get_default_features(self) -> Dict:
        """
        Return default features for profiles without metadata
        """
        return {
            'origin': 'Unknown',
            'process': 'Unknown',
            'roast_level': 'Medium',
            'variety': 'Unknown',
            'target_finish_temp': 395,
        }


class PreprocessedDataLoader:
    """
    Load preprocessed train/val data and create PyTorch DataLoaders
    """

    def __init__(self, preprocessed_dir: str = "preprocessed_data"):
        """
        Args:
            preprocessed_dir: Directory containing train_profiles.json, etc.
        """
        self.preprocessed_dir = Path(preprocessed_dir)

        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed directory not found: {preprocessed_dir}\n"
                "Please run: python prepare_training_data.py"
            )

        # Initialize encoders and vocab
        self.encoders = {
            'origin': LabelEncoder(),
            'process': LabelEncoder(),
            'roast_level': LabelEncoder(),
            'variety': LabelEncoder()
        }
        self.flavor_vocab = {}

        # Load data
        self.train_profiles = None
        self.val_profiles = None
        self.train_metadata = None
        self.val_metadata = None

    def load_data(self):
        """Load train and val profiles from JSON"""

        print("\n" + "="*80)
        print("LOADING PREPROCESSED DATA")
        print("="*80)

        # Load profiles
        train_path = self.preprocessed_dir / "train_profiles.json"
        val_path = self.preprocessed_dir / "val_profiles.json"

        with open(train_path, 'r') as f:
            self.train_profiles = json.load(f)
        print(f"✓ Loaded {len(self.train_profiles)} training profiles")

        with open(val_path, 'r') as f:
            self.val_profiles = json.load(f)
        print(f"✓ Loaded {len(self.val_profiles)} validation profiles")

        # Load metadata
        self.train_metadata = pd.read_csv(self.preprocessed_dir / "train_metadata.csv")
        self.val_metadata = pd.read_csv(self.preprocessed_dir / "val_metadata.csv")
        print(f"✓ Loaded metadata")

        # Build encoders
        self._build_encoders()

        print("="*80 + "\n")

        return self.train_profiles, self.val_profiles

    def _build_encoders(self):
        """Build label encoders from training data"""

        # Combine train and val metadata for complete vocabulary
        all_metadata = pd.concat([self.train_metadata, self.val_metadata])

        # Fit encoders
        self.encoders['origin'].fit(all_metadata['origin'].fillna('Unknown').unique())
        self.encoders['process'].fit(all_metadata['process'].fillna('Unknown').unique())
        self.encoders['roast_level'].fit(all_metadata['roast_level'].fillna('Medium').unique())
        self.encoders['variety'].fit(all_metadata['variety'].fillna('Unknown').unique())

        # Build flavor vocabulary
        all_flavors = set()
        for flavor_str in all_metadata['flavor_notes'].dropna():
            if isinstance(flavor_str, str):
                flavors = [f.strip().lower() for f in flavor_str.split(',')]
                all_flavors.update(flavors)

        self.flavor_vocab = {flavor: idx for idx, flavor in enumerate(sorted(all_flavors))}

        print(f"\n📊 Feature Vocabulary:")
        print(f"   Origins: {len(self.encoders['origin'].classes_)}")
        print(f"   Processes: {len(self.encoders['process'].classes_)}")
        print(f"   Roast Levels: {len(self.encoders['roast_level'].classes_)}")
        print(f"   Varieties: {len(self.encoders['variety'].classes_)}")
        print(f"   Flavors: {len(self.flavor_vocab)}")

    def create_dataloaders(
        self,
        batch_size: int = 16,
        max_sequence_length: int = 1000,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train and val

        Args:
            batch_size: Batch size
            max_sequence_length: Max temperature sequence length
            num_workers: Number of data loading workers

        Returns:
            (train_loader, val_loader)
        """
        if self.train_profiles is None:
            raise ValueError("Must call load_data() first!")

        print(f"\n📦 Creating DataLoaders:")
        print(f"   Batch size: {batch_size}")
        print(f"   Max sequence length: {max_sequence_length}")

        # Create datasets
        train_dataset = PreprocessedRoastDataset(
            self.train_profiles,
            self.train_metadata,
            self.encoders,
            self.flavor_vocab,
            max_sequence_length
        )

        val_dataset = PreprocessedRoastDataset(
            self.val_profiles,
            self.val_metadata,
            self.encoders,
            self.flavor_vocab,
            max_sequence_length
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches\n")

        return train_loader, val_loader

    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions for model initialization

        Returns:
            Dict with vocabulary sizes
        """
        return {
            'num_origins': len(self.encoders['origin'].classes_),
            'num_processes': len(self.encoders['process'].classes_),
            'num_roast_levels': len(self.encoders['roast_level'].classes_),
            'num_varieties': len(self.encoders['variety'].classes_),
            'num_flavors': len(self.flavor_vocab),
            'num_continuous': 3  # target_temp, altitude, bean_density
        }


# Usage example
if __name__ == "__main__":
    print("="*80)
    print("TESTING PREPROCESSED DATA LOADER")
    print("="*80)

    # Initialize loader
    loader = PreprocessedDataLoader(preprocessed_dir="preprocessed_data")

    # Load data
    train_profiles, val_profiles = loader.load_data()

    # Create dataloaders
    train_loader, val_loader = loader.create_dataloaders(
        batch_size=4,  # Small batch for testing
        max_sequence_length=1000
    )

    # Get feature dimensions
    dims = loader.get_feature_dimensions()
    print("📏 Feature Dimensions:")
    for key, value in dims.items():
        print(f"   {key}: {value}")

    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    for batch in train_loader:
        print(f"   Temperatures shape: {batch['temperatures'].shape}")
        print(f"   Mask shape: {batch['mask'].shape}")
        print(f"   Origin indices: {batch['features']['categorical']['origin'].shape}")
        print(f"   Continuous features: {batch['features']['continuous']['target_finish_temp'].shape}")
        print(f"   Flavor vector: {batch['features']['flavors'].shape}")
        print(f"   Product names: {batch['product_name'][:2]}")
        break

    print("\n✅ DATA LOADER WORKING CORRECTLY!")
    print("="*80)
