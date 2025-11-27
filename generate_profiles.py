"""
Profile Generation Script for RoastFormer

Generate custom roast profiles by specifying:
- Bean characteristics (origin, process, variety)
- Target roast level
- Desired flavor profile
- Roasting parameters (start temp, duration)

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt

# Import data loader
from src.dataset.preprocessed_data_loader import PreprocessedDataLoader

# Import adapted transformer
from src.model.transformer_adapter import (
    AdaptedConditioningModule,
    AdaptedRoastFormer
)


class ProfileGenerator:
    """
    Generate custom roast profiles using trained RoastFormer
    """

    def __init__(self, checkpoint_path: str, preprocessed_dir: str = 'preprocessed_data', device: str = 'cpu'):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        print(f"\n📦 Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint['config']
        self.feature_dims = self.checkpoint['feature_dims']

        print(f"✓ Checkpoint loaded (Epoch {self.checkpoint['epoch']})")
        print(f"✓ Best val loss: {self.checkpoint['best_val_loss']:.4f}")

        # Load data to get encoders
        self.data_loader = PreprocessedDataLoader(preprocessed_dir=preprocessed_dir)
        self.data_loader.load_data()

        # Initialize model
        self.model = self._initialize_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model initialized and ready for generation")

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

    def list_available_features(self):
        """Print all available feature values"""
        print("\n" + "="*80)
        print("AVAILABLE FEATURE VALUES")
        print("="*80)

        print("\n📍 Origins:")
        for i, origin in enumerate(self.data_loader.encoders['origin'].classes_):
            print(f"   {i}: {origin}")

        print("\n⚙️  Processes:")
        for i, process in enumerate(self.data_loader.encoders['process'].classes_):
            print(f"   {i}: {process}")

        print("\n🔥 Roast Levels:")
        for i, level in enumerate(self.data_loader.encoders['roast_level'].classes_):
            print(f"   {i}: {level}")

        print("\n🌱 Varieties:")
        for i, variety in enumerate(self.data_loader.encoders['variety'].classes_):
            print(f"   {i}: {variety}")

        print(f"\n🎨 Flavors: {len(self.data_loader.flavor_vocab)} unique flavors")
        print("   Sample flavors:", list(self.data_loader.flavor_vocab.keys())[:10])
        print("   (Use comma-separated list, e.g., 'berries,chocolate,floral')")

        print("="*80)

    def create_features_from_spec(self, spec: Dict) -> Dict:
        """
        Create feature dict from user specification

        Args:
            spec: Dict with keys:
                - origin: str (e.g., 'Ethiopia')
                - process: str (e.g., 'Washed')
                - roast_level: str (e.g., 'Light')
                - variety: str (e.g., 'Heirloom')
                - flavors: List[str] (e.g., ['berries', 'floral'])
                - target_temp: float (e.g., 395.0)
                - altitude: float (e.g., 2000.0)
                - bean_density: float (e.g., 0.72)

        Returns:
            Feature dict compatible with model
        """
        # Encode categorical features
        categorical = {}

        # Origin
        if spec['origin'] in self.data_loader.encoders['origin'].classes_:
            categorical['origin'] = torch.LongTensor([
                self.data_loader.encoders['origin'].transform([spec['origin']])[0]
            ]).unsqueeze(0)
        else:
            print(f"⚠️  Unknown origin '{spec['origin']}', using default")
            categorical['origin'] = torch.LongTensor([[0]])

        # Process
        if spec['process'] in self.data_loader.encoders['process'].classes_:
            categorical['process'] = torch.LongTensor([
                self.data_loader.encoders['process'].transform([spec['process']])[0]
            ]).unsqueeze(0)
        else:
            print(f"⚠️  Unknown process '{spec['process']}', using default")
            categorical['process'] = torch.LongTensor([[0]])

        # Roast level
        if spec['roast_level'] in self.data_loader.encoders['roast_level'].classes_:
            categorical['roast_level'] = torch.LongTensor([
                self.data_loader.encoders['roast_level'].transform([spec['roast_level']])[0]
            ]).unsqueeze(0)
        else:
            print(f"⚠️  Unknown roast level '{spec['roast_level']}', using default")
            categorical['roast_level'] = torch.LongTensor([[0]])

        # Variety
        if spec['variety'] in self.data_loader.encoders['variety'].classes_:
            categorical['variety'] = torch.LongTensor([
                self.data_loader.encoders['variety'].transform([spec['variety']])[0]
            ]).unsqueeze(0)
        else:
            print(f"⚠️  Unknown variety '{spec['variety']}', using default")
            categorical['variety'] = torch.LongTensor([[0]])

        # Continuous features (normalized)
        continuous = {
            'target_finish_temp': torch.FloatTensor([[spec['target_temp'] / 425.0]]),
            'altitude': torch.FloatTensor([[spec['altitude'] / 2500.0]]),
            'bean_density': torch.FloatTensor([[spec['bean_density'] / 0.80]])
        }

        # Flavors (multi-hot encoding)
        flavor_vector = torch.zeros(1, len(self.data_loader.flavor_vocab))
        for flavor in spec['flavors']:
            flavor_lower = flavor.lower().strip()
            if flavor_lower in self.data_loader.flavor_vocab:
                idx = self.data_loader.flavor_vocab[flavor_lower]
                flavor_vector[0, idx] = 1.0
            else:
                print(f"⚠️  Unknown flavor '{flavor}', skipping")

        features = {
            'categorical': categorical,
            'continuous': continuous,
            'flavors': flavor_vector
        }

        return features

    def generate_profile(
        self,
        spec: Dict,
        start_temp: float = 426.0,
        target_duration: int = 600
    ) -> Dict:
        """
        Generate roast profile from specification

        Args:
            spec: Coffee specification dict
            start_temp: Charge temperature (°F)
            target_duration: Profile duration (seconds)

        Returns:
            Dict with generated profile and metadata
        """
        print(f"\n🔥 Generating roast profile...")
        print(f"   Origin: {spec['origin']}")
        print(f"   Process: {spec['process']}")
        print(f"   Roast Level: {spec['roast_level']}")
        print(f"   Variety: {spec['variety']}")
        print(f"   Flavors: {', '.join(spec['flavors'])}")
        print(f"   Target Temp: {spec['target_temp']}°F")
        print(f"   Start Temp: {start_temp}°F")
        print(f"   Duration: {target_duration}s ({target_duration/60:.1f} min)")

        # Create features
        features = self.create_features_from_spec(spec)

        # Move features to device
        features = {
            'categorical': {k: v.to(self.device) for k, v in features['categorical'].items()},
            'continuous': {k: v.to(self.device) for k, v in features['continuous'].items()},
            'flavors': features['flavors'].to(self.device)
        }

        # Generate profile
        with torch.no_grad():
            generated = self.model.generate(
                features=features,
                start_temp=start_temp,
                target_duration=target_duration,
                device=self.device
            )

        # Compute metrics
        ror = np.diff(generated) * 60  # Rate of rise (°F/min)
        turning_idx = np.argmin(generated)

        result = {
            'specification': spec,
            'parameters': {
                'start_temp': start_temp,
                'target_duration': target_duration
            },
            'profile': {
                'temperatures': generated.tolist(),
                'duration_seconds': len(generated),
                'duration_minutes': len(generated) / 60
            },
            'metrics': {
                'start_temp': float(generated[0]),
                'final_temp': float(generated[-1]),
                'min_temp': float(generated.min()),
                'max_temp': float(generated.max()),
                'turning_point_idx': int(turning_idx),
                'turning_point_temp': float(generated[turning_idx]),
                'mean_ror': float(ror.mean()),
                'std_ror': float(ror.std()),
                'min_ror': float(ror.min()),
                'max_ror': float(ror.max())
            }
        }

        print(f"\n✓ Profile generated!")
        print(f"   Final temp: {result['metrics']['final_temp']:.1f}°F")
        print(f"   Turning point: {result['metrics']['turning_point_temp']:.1f}°F at {turning_idx}s")
        print(f"   Mean RoR: {result['metrics']['mean_ror']:.1f}°F/min")

        return result

    def plot_profile(self, result: Dict, output_path: str = None):
        """Create visualization of generated profile"""
        temps = np.array(result['profile']['temperatures'])
        ror = np.diff(temps) * 60

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Temperature profile
        ax1.plot(temps, linewidth=2, color='#2E86AB')
        ax1.axhline(y=result['specification']['target_temp'],
                    color='red', linestyle='--', alpha=0.5,
                    label=f"Target: {result['specification']['target_temp']}°F")
        ax1.axvline(x=result['metrics']['turning_point_idx'],
                    color='orange', linestyle=':', alpha=0.5,
                    label='Turning Point')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Temperature (°F)')
        ax1.set_title(
            f"Generated Roast Profile\n"
            f"{result['specification']['origin']} - {result['specification']['process']} - "
            f"{result['specification']['roast_level']}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rate of Rise
        ax2.plot(ror, linewidth=2, color='#A23B72')
        ax2.axhline(y=20, color='red', linestyle=':', alpha=0.5, label='Min RoR (20°F/min)')
        ax2.axhline(y=100, color='red', linestyle=':', alpha=0.5, label='Max RoR (100°F/min)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Rate of Rise (°F/min)')
        ax2.set_title('Rate of Rise (RoR)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"✓ Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate roast profiles with RoastFormer')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_transformer_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--list_features', action='store_true',
                        help='List all available feature values')

    # Coffee specification
    parser.add_argument('--origin', type=str, default='Ethiopia',
                        help='Coffee origin')
    parser.add_argument('--process', type=str, default='Washed',
                        help='Processing method')
    parser.add_argument('--roast_level', type=str, default='Light',
                        help='Target roast level')
    parser.add_argument('--variety', type=str, default='Heirloom',
                        help='Coffee variety')
    parser.add_argument('--flavors', type=str, default='berries,floral,citrus',
                        help='Comma-separated flavor notes')
    parser.add_argument('--target_temp', type=float, default=395.0,
                        help='Target finish temperature (°F)')
    parser.add_argument('--altitude', type=float, default=2000.0,
                        help='Growing altitude (MASL)')
    parser.add_argument('--bean_density', type=float, default=0.72,
                        help='Bean density proxy (g/cm³)')

    # Roasting parameters
    parser.add_argument('--start_temp', type=float, default=426.0,
                        help='Charge temperature (°F)')
    parser.add_argument('--duration', type=int, default=600,
                        help='Target roast duration (seconds)')

    # Output options
    parser.add_argument('--output', type=str, default='results/generated_profiles',
                        help='Output directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print("="*80)
    print("ROASTFORMER PROFILE GENERATOR")
    print("="*80)

    # Initialize generator
    generator = ProfileGenerator(args.checkpoint, device=args.device)

    # List features if requested
    if args.list_features:
        generator.list_available_features()
        return

    # Create specification
    spec = {
        'origin': args.origin,
        'process': args.process,
        'roast_level': args.roast_level,
        'variety': args.variety,
        'flavors': [f.strip() for f in args.flavors.split(',')],
        'target_temp': args.target_temp,
        'altitude': args.altitude,
        'bean_density': args.bean_density
    }

    # Generate profile
    result = generator.generate_profile(
        spec=spec,
        start_temp=args.start_temp,
        target_duration=args.duration
    )

    # Save result
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"profile_{spec['origin']}_{spec['process']}_{spec['roast_level']}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Profile saved to {output_file}")

    # Create plot if requested
    if args.plot:
        plot_file = output_dir / f"profile_{spec['origin']}_{spec['process']}_{spec['roast_level']}.png"
        generator.plot_profile(result, str(plot_file))

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
