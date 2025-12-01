"""
Preprocessed Data Loader for RoastFormer Training (WITH TEMPERATURE NORMALIZATION)

CRITICAL FIX: This version normalizes temperatures to [0, 1] range before training.
This fixes the fundamental bug where models predicted ~5-10°F instead of 150-450°F.

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
Date: November 19, 2024
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
# TEMPERATURE NORMALIZATION CONSTANTS
# ═══════════════════════════════════════════════════════════════
# These define the range for normalizing temperatures to [0, 1]

TEMP_MIN = 100.0  # Minimum possible roast temperature (°F)
TEMP_MAX = 500.0  # Maximum possible roast temperature (°F)

# Why these values?
# - Roast profiles typically range from 250-450°F
# - We use wider range (100-500°F) to avoid edge effects
# - This maps temperatures to [0, 1]: normalized = (temp - 100) / 400

def normalize_temperature(temp: float) -> float:
    """
    Normalize temperature from °F to [0, 1] range

    Args:
        temp: Temperature in °F (typically 150-450)

    Returns:
        Normalized temperature in [0, 1] (e.g., 275°F → 0.4375)
    """
    return (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)


def denormalize_temperature(temp_norm: float) -> float:
    """
    Denormalize temperature from [0, 1] range back to °F

    Args:
        temp_norm: Normalized temperature in [0, 1]

    Returns:
        Temperature in °F (e.g., 0.4375 → 275°F)
    """
    return temp_norm * (TEMP_MAX - TEMP_MIN) + TEMP_MIN


class PreprocessedRoastDataset(Dataset):
    """
    PyTorch Dataset for preprocessed roast profiles

    NEW: Normalizes temperatures to [0, 1] range for stable training
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

        NEW: Temperatures are normalized to [0, 1] range

        Returns:
            temps: NORMALIZED temperature sequence (padded/truncated to max_length)
            features: Dict of encoded features
            mask: Boolean mask for valid timesteps
        """
        profile = self.profiles[idx]

        # Extract temperature sequence (raw °F)
        bean_temp_data = profile['roast_profile']['bean_temp']
        temps_raw = np.array([point['value'] for point in bean_temp_data])

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL FIX: Normalize temperatures to [0, 1]
        # ═══════════════════════════════════════════════════════════════
        temps = np.array([normalize_temperature(t) for t in temps_raw])

        # Example: 425°F → (425 - 100) / 400 = 0.8125
        # Now model predicts in range [0, 1] instead of [150, 450]

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
        temps_tensor = torch.FloatTensor(temps)  # Now in [0, 1] range!
        mask_tensor = torch.BoolTensor(mask)

        return {
            'temperatures': temps_tensor,  # NORMALIZED
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

    NEW: Returns normalized temperatures [0, 1]
    """

    def __init__(self, preprocessed_dir: str = "preprocessed_data"):
        """
        Args:
            preprocessed_dir: Directory containing train_profiles.json, etc.
        """
        self.preprocessed_dir = Path(preprocessed_dir)

        self.train_profiles = None
        self.val_profiles = None
        self.train_metadata = None
        self.val_metadata = None
        self.metadata_df = None

        # Initialize encoders
        self.encoders = {
            'origin': LabelEncoder(),
            'process': LabelEncoder(),
            'roast_level': LabelEncoder(),
            'variety': LabelEncoder()
        }
        self.flavor_vocab = {}

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load preprocessed profiles and metadata

        Returns:
            (train_profiles, val_profiles)
        """
        print("\n" + "="*80)
        print("LOADING PREPROCESSED DATA (WITH NORMALIZATION)")
        print("="*80)

        # Load training profiles
        train_path = self.preprocessed_dir / 'train_profiles.json'
        with open(train_path, 'r') as f:
            self.train_profiles = json.load(f)
        print(f"✓ Loaded {len(self.train_profiles)} training profiles")

        # Load validation profiles
        val_path = self.preprocessed_dir / 'val_profiles.json'
        with open(val_path, 'r') as f:
            self.val_profiles = json.load(f)
        print(f"✓ Loaded {len(self.val_profiles)} validation profiles")

        # Load metadata (combine train and val metadata)
        train_meta_path = self.preprocessed_dir / 'train_metadata.csv'
        val_meta_path = self.preprocessed_dir / 'val_metadata.csv'

        self.train_metadata = pd.read_csv(train_meta_path)
        self.val_metadata = pd.read_csv(val_meta_path)
        self.metadata_df = pd.concat([self.train_metadata, self.val_metadata], ignore_index=True)
        print(f"✓ Loaded metadata")

        # Build encoders from metadata
        self._build_encoders_and_vocab()

        print(f"\n📊 Feature Vocabulary:")
        print(f"   Origins: {len(self.encoders['origin'].classes_)}")
        print(f"   Processes: {len(self.encoders['process'].classes_)}")
        print(f"   Roast Levels: {len(self.encoders['roast_level'].classes_)}")
        print(f"   Varieties: {len(self.encoders['variety'].classes_)}")
        print(f"   Flavors: {len(self.flavor_vocab)}")

        print(f"\n🔧 Temperature Normalization:")
        print(f"   Range: [{TEMP_MIN}°F, {TEMP_MAX}°F] → [0, 1]")
        print(f"   Example: 275°F → {normalize_temperature(275):.4f}")
        print(f"   Example: 425°F → {normalize_temperature(425):.4f}")

        print("="*80 + "\n")

        return self.train_profiles, self.val_profiles

    def _build_encoders_and_vocab(self):
        """Build label encoders and flavor vocabulary from metadata"""

        # Fit label encoders on all metadata (train + val combined)
        for key in ['origin', 'process', 'roast_level', 'variety']:
            if key in self.metadata_df.columns:
                # Get unique values, excluding NaN
                values = self.metadata_df[key].dropna().unique()
                self.encoders[key].fit(values)

        # Build flavor vocabulary
        all_flavors = set()

        # Process both train and val profiles for flavors
        for profile in self.train_profiles + self.val_profiles:
            metadata = profile.get('metadata', {})
            flavor_notes_parsed = metadata.get('flavor_notes_parsed', [])

            # Handle both list and string formats
            if isinstance(flavor_notes_parsed, list):
                flavors = [f.strip().lower() for f in flavor_notes_parsed if f]
            elif isinstance(flavor_notes_parsed, str):
                flavors = [f.strip().lower() for f in flavor_notes_parsed.split(',') if f.strip()]
            else:
                flavors = []

            all_flavors.update(flavors)

        self.flavor_vocab = {flavor: idx for idx, flavor in enumerate(sorted(all_flavors))}

    def create_dataloaders(
        self,
        batch_size: int = 16,
        max_sequence_length: int = 1000,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders

        Args:
            batch_size: Batch size
            max_sequence_length: Max temperature sequence length
            num_workers: Number of data loading workers

        Returns:
            (train_loader, val_loader)
        """
        if self.train_profiles is None:
            raise ValueError("Call load_data() first")

        print(f"📦 Creating DataLoaders:")
        print(f"   Batch size: {batch_size}")
        print(f"   Max sequence length: {max_sequence_length}")

        # Create datasets
        train_dataset = PreprocessedRoastDataset(
            profiles=self.train_profiles,
            metadata_df=self.metadata_df,
            encoders=self.encoders,
            flavor_vocab=self.flavor_vocab,
            max_sequence_length=max_sequence_length
        )

        val_dataset = PreprocessedRoastDataset(
            profiles=self.val_profiles,
            metadata_df=self.metadata_df,
            encoders=self.encoders,
            flavor_vocab=self.flavor_vocab,
            max_sequence_length=max_sequence_length
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches\n")

        return train_loader, val_loader

    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions for model initialization

        Returns:
            Dict with feature dimension sizes
        """
        return {
            'num_origins': len(self.encoders['origin'].classes_),
            'num_processes': len(self.encoders['process'].classes_),
            'num_roast_levels': len(self.encoders['roast_level'].classes_),
            'num_varieties': len(self.encoders['variety'].classes_),
            'num_flavors': len(self.flavor_vocab),
            'num_continuous': 3  # target_finish_temp, altitude, bean_density
        }


# ═══════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test normalization functions
    print("="*80)
    print("TESTING TEMPERATURE NORMALIZATION")
    print("="*80)

    test_temps = [150, 250, 350, 425, 450]
    print("\nForward (°F → normalized):")
    for temp in test_temps:
        norm = normalize_temperature(temp)
        print(f"  {temp}°F → {norm:.4f}")

    print("\nReverse (normalized → °F):")
    test_norms = [0.0, 0.25, 0.5, 0.75, 1.0]
    for norm in test_norms:
        temp = denormalize_temperature(norm)
        print(f"  {norm:.2f} → {temp:.1f}°F")

    print("\nRound-trip test:")
    for temp in [275, 350, 425]:
        norm = normalize_temperature(temp)
        back = denormalize_temperature(norm)
        print(f"  {temp}°F → {norm:.4f} → {back:.1f}°F ✓")

    print("\n" + "="*80)
