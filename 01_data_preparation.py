"""
Phase 1: Data Preparation for RoastFormer
Load Onyx profiles and prepare for transformer training
"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class RoastProfileDataLoader:
    """
    Prepare Onyx Coffee Lab profiles for transformer training
    Auto-discovers all onyx_dataset_* directories
    """
    
    def __init__(self, dataset_dir=None, auto_discover=True):
        """
        Args:
            dataset_dir: Specific directory, or None for auto-discovery
            auto_discover: If True, finds all onyx_dataset_* directories
        """
        self.dataset_dirs = []
        self.profiles = []
        self.metadata_df = None
        
        # Feature encoders (will be fit on data)
        self.origin_encoder = LabelEncoder()
        self.process_encoder = LabelEncoder()
        self.roast_level_encoder = LabelEncoder()
        self.variety_encoder = LabelEncoder()
        
        # Flavor vocabulary
        self.flavor_vocab = {}
        
        if auto_discover:
            self._discover_datasets()
        elif dataset_dir:
            self.dataset_dirs = [Path(dataset_dir)]
            print(f"📁 Using specific dataset: {dataset_dir}")
        else:
            raise ValueError("Must provide dataset_dir or use auto_discover=True")
    
    def _discover_datasets(self):
        """Auto-discover all onyx_dataset_* directories"""
        import glob
        
        # Find all onyx_dataset_* directories
        pattern = 'onyx_dataset_*'
        found_dirs = glob.glob(pattern)
        
        # Filter to only directories
        found_dirs = [d for d in found_dirs if Path(d).is_dir()]
        
        if not found_dirs:
            raise FileNotFoundError(
                "No onyx_dataset_* directories found!\n"
                "Please run: python onyx_dataset_builder_v3.1_ADDITIVE_FINAL.py"
            )
        
        # Sort by date (most recent first)
        found_dirs.sort(reverse=True)
        
        self.dataset_dirs = [Path(d) for d in found_dirs]
        
        print(f"📁 Auto-discovered {len(self.dataset_dirs)} dataset(s):")
        for d in self.dataset_dirs:
            print(f"   • {d.name}")
        print()
    
    def load_dataset(self):
        """Load all profiles and metadata from all discovered datasets"""
        
        all_metadata = []
        
        for dataset_dir in self.dataset_dirs:
            print(f"\n📂 Loading from: {dataset_dir.name}")
            
            # Load summary CSV
            csv_path = dataset_dir / 'dataset_summary.csv'
            if not csv_path.exists():
                print(f"   ⚠️  No CSV found, skipping...")
                continue
            
            metadata = pd.read_csv(csv_path)
            metadata['source_dataset'] = dataset_dir.name  # Track source
            all_metadata.append(metadata)
            print(f"   ✓ {len(metadata)} products")
            
            # Load individual profile JSONs
            profiles_dir = dataset_dir / 'profiles'
            if not profiles_dir.exists():
                print(f"   ⚠️  No profiles/ directory, skipping...")
                continue
            
            profile_count = 0
            for profile_file in profiles_dir.glob('*.json'):
                try:
                    with open(profile_file, 'r') as f:
                        profile = json.load(f)
                        profile['source_dataset'] = dataset_dir.name
                        self.profiles.append(profile)
                        profile_count += 1
                except Exception as e:
                    print(f"   ⚠️  Error loading {profile_file.name}: {e}")
            
            print(f"   ✓ {profile_count} profiles")
        
        if not all_metadata:
            raise FileNotFoundError(
                "No valid dataset_summary.csv found in any dataset directory!"
            )
        
        # Combine all metadata
        self.metadata_df = pd.concat(all_metadata, ignore_index=True)
        
        # Remove duplicates (same product, same batch)
        if 'roast_info_batch' in self.metadata_df.columns:
            before = len(self.metadata_df)
            self.metadata_df = self.metadata_df.drop_duplicates(
                subset=['product_name', 'roast_info_batch'],
                keep='last'  # Keep most recent
            )
            after = len(self.metadata_df)
            if before > after:
                print(f"\n🔄 Removed {before - after} duplicate batches")
        
        print(f"\n✅ TOTAL LOADED:")
        print(f"   • Metadata: {len(self.metadata_df)} unique products")
        print(f"   • Profiles: {len(self.profiles)} roast curves")
        
        # Build feature encoders
        self._build_encoders()
        
        return self.profiles, self.metadata_df
    
    def _build_encoders(self):
        """Build label encoders for categorical features"""
        
        # Fit encoders on unique values
        self.origin_encoder.fit(self.metadata_df['origin'].dropna().unique())
        self.process_encoder.fit(self.metadata_df['process'].dropna().unique())
        self.roast_level_encoder.fit(self.metadata_df['roast_level'].dropna().unique())
        self.variety_encoder.fit(self.metadata_df['variety'].dropna().unique())
        
        # Build flavor vocabulary
        all_flavors = set()
        for flavors_str in self.metadata_df['flavor_notes_parsed'].dropna():
            if isinstance(flavors_str, str):
                flavors = [f.strip() for f in flavors_str.split(',')]
                all_flavors.update([f.lower() for f in flavors])
        
        self.flavor_vocab = {flavor: idx for idx, flavor in enumerate(sorted(all_flavors))}
        
        print(f"\n📊 Feature Vocabulary:")
        print(f"   Origins: {len(self.origin_encoder.classes_)}")
        print(f"   Processes: {len(self.process_encoder.classes_)}")
        print(f"   Roast Levels: {len(self.roast_level_encoder.classes_)}")
        print(f"   Varieties: {len(self.variety_encoder.classes_)}")
        print(f"   Flavors: {len(self.flavor_vocab)}")
    
    def prepare_training_data(self, max_sequence_length=1000):
        """
        Convert profiles to training format
        
        Returns:
            List of (temperature_sequence, features_dict) tuples
        """
        training_data = []
        
        for profile in self.profiles:
            # Extract temperature sequence
            bean_temps = profile['roast_profile']['bean_temp']
            temps = np.array([point['value'] for point in bean_temps])
            
            # Truncate or pad to max_sequence_length
            if len(temps) > max_sequence_length:
                temps = temps[:max_sequence_length]
            
            # Extract metadata
            metadata = profile['metadata']
            product_name = metadata['product_name']
            
            # Find matching row in metadata_df
            row = self.metadata_df[self.metadata_df['product_name'] == product_name]
            
            if row.empty:
                print(f"Warning: No metadata for {product_name}")
                continue
            
            row = row.iloc[0]
            
            # Encode categorical features
            features = {
                'origin': self.origin_encoder.transform([row['origin']])[0] if pd.notna(row['origin']) else 0,
                'process': self.process_encoder.transform([row['process']])[0] if pd.notna(row['process']) else 0,
                'roast_level': self.roast_level_encoder.transform([row['roast_level']])[0] if pd.notna(row['roast_level']) else 0,
                'variety': self.variety_encoder.transform([row['variety']])[0] if pd.notna(row['variety']) else 0,
                
                # Continuous features (normalized)
                'target_finish_temp': row['target_finish_temp'] / 425.0 if pd.notna(row['target_finish_temp']) else 0.93,
                'altitude': row['altitude_numeric'] / 2500.0 if pd.notna(row['altitude_numeric']) else 0.6,
                'bean_density': row['bean_density_proxy'] / 0.80 if pd.notna(row['bean_density_proxy']) else 0.85,
                
                # Flavor encoding (one-hot)
                'flavors': self._encode_flavors(row['flavor_notes_parsed']) if pd.notna(row['flavor_notes_parsed']) else torch.zeros(len(self.flavor_vocab)),
                
                # Raw data (for reference)
                'product_name': product_name,
                'duration': len(temps)
            }
            
            training_data.append((temps, features))
        
        print(f"\n✓ Prepared {len(training_data)} training examples")
        return training_data
    
    def _encode_flavors(self, flavors_str):
        """One-hot encode flavor notes"""
        flavor_vector = torch.zeros(len(self.flavor_vocab))
        
        if isinstance(flavors_str, str):
            flavors = [f.strip().lower() for f in flavors_str.split(',')]
            for flavor in flavors:
                if flavor in self.flavor_vocab:
                    idx = self.flavor_vocab[flavor]
                    flavor_vector[idx] = 1.0
        
        return flavor_vector
    
    def create_train_val_split(self, training_data, val_ratio=0.2, random_state=42):
        """Split data into train and validation sets"""
        
        train_data, val_data = train_test_split(
            training_data,
            test_size=val_ratio,
            random_state=random_state
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(train_data)} profiles")
        print(f"   Validation: {len(val_data)} profiles")
        
        # Warning for small datasets
        if len(train_data) < 20:
            print(f"\n⚠️  WARNING: Small training set ({len(train_data)} samples)")
            print(f"   Risk of overfitting. Consider:")
            print(f"   1. Collecting more data")
            print(f"   2. Using smaller model")
            print(f"   3. Heavy regularization (dropout, weight decay)")
        
        return train_data, val_data
    
    def save_preprocessed_data(self, train_data, val_data, output_dir='preprocessed_data'):
        """Save preprocessed data for quick loading"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save data
        torch.save({
            'train_data': train_data,
            'val_data': val_data,
            'origin_encoder': self.origin_encoder,
            'process_encoder': self.process_encoder,
            'roast_level_encoder': self.roast_level_encoder,
            'variety_encoder': self.variety_encoder,
            'flavor_vocab': self.flavor_vocab
        }, output_path / 'training_data.pt')
        
        print(f"\n✓ Saved preprocessed data to: {output_path}")
        
        # Save statistics
        stats = {
            'num_train': len(train_data),
            'num_val': len(val_data),
            'num_origins': len(self.origin_encoder.classes_),
            'num_processes': len(self.process_encoder.classes_),
            'num_roast_levels': len(self.roast_level_encoder.classes_),
            'num_varieties': len(self.variety_encoder.classes_),
            'num_flavors': len(self.flavor_vocab),
            'max_sequence_length': max([len(temps) for temps, _ in train_data + val_data])
        }
        
        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


# Usage example
if __name__ == "__main__":
    print("="*60)
    print("ROASTFORMER DATA PREPARATION")
    print("="*60)
    
    # Initialize loader (auto-discovers all onyx_dataset_* directories)
    loader = RoastProfileDataLoader(auto_discover=True)
    
    # Load dataset
    profiles, metadata = loader.load_dataset()
    
    # Prepare training data
    training_data = loader.prepare_training_data(max_sequence_length=1000)
    
    # Create train/val split
    train_data, val_data = loader.create_train_val_split(training_data, val_ratio=0.2)
    
    # Save for later
    stats = loader.save_preprocessed_data(train_data, val_data)
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Ready for transformer training!")
    print(f"\n📊 Dataset Summary:")
    print(f"   Total profiles discovered: {len(profiles)}")
    print(f"   Train examples: {stats['num_train']}")
    print(f"   Val examples: {stats['num_val']}")
    print(f"\n🎯 Feature Dimensions:")
    print(f"   Origins: {stats['num_origins']}")
    print(f"   Processes: {stats['num_processes']}")
    print(f"   Roast Levels: {stats['num_roast_levels']}")
    print(f"   Varieties: {stats['num_varieties']}")
    print(f"   Flavors: {stats['num_flavors']}")
    print(f"   Continuous: 3")
    print(f"\n💾 Saved to: preprocessed_data/")
    print("="*60)