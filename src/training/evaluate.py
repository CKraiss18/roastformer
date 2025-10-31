"""
Evaluation Pipeline for RoastFormer

Load trained models and evaluate on test data:
- Generate roast profiles from conditioning
- Compute comprehensive metrics
- Create comparison visualizations
- Save results

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.roastformer import RoastFormer
from src.utils.metrics import comprehensive_evaluation, evaluate_batch, compute_success_metrics
from src.utils.visualization import plot_comparison, plot_batch_profiles, save_profile_visualization
from src.utils.validation import RoastProfileValidator


class RoastFormerEvaluator:
    """
    Evaluation pipeline for trained RoastFormer models
    """

    def __init__(
        self,
        model: RoastFormer,
        checkpoint_path: str,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator

        Args:
            model: RoastFormer model (architecture only)
            checkpoint_path: Path to trained checkpoint
            device: Device to run on
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.config = checkpoint.get('config', {})
        self.epoch = checkpoint.get('epoch', 0)

        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"  Trained for {self.epoch} epochs")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    def generate_profile(
        self,
        conditioning: torch.Tensor,
        start_temp: float = 426.0,
        num_steps: int = 600,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate a roast profile from conditioning

        Args:
            conditioning: Conditioning features tensor
            start_temp: Starting temperature (°F)
            num_steps: Number of time steps to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated temperature sequence
        """
        self.model.eval()

        with torch.no_grad():
            # Initialize with start temperature
            current_seq = torch.tensor([[start_temp]], device=self.device)

            generated_temps = [start_temp]

            for step in range(num_steps - 1):
                # Forward pass
                output = self.model(current_seq, conditioning.unsqueeze(0))

                # Get next temperature
                next_temp = output[0, -1, 0].item()

                # Apply temperature sampling if desired
                if temperature != 1.0:
                    next_temp = next_temp / temperature

                generated_temps.append(next_temp)

                # Append to sequence
                current_seq = torch.cat([
                    current_seq,
                    torch.tensor([[next_temp]], device=self.device)
                ], dim=1)

        return np.array(generated_temps)

    def evaluate_on_dataset(
        self,
        test_profiles: List[Dict],
        output_dir: str,
        num_samples: Optional[int] = None
    ) -> Dict:
        """
        Evaluate model on a test dataset

        Args:
            test_profiles: List of profile dicts
            output_dir: Directory to save results
            num_samples: Optional limit on number of profiles to evaluate

        Returns:
            Evaluation results dict
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if num_samples:
            test_profiles = test_profiles[:num_samples]

        print(f"\nEvaluating on {len(test_profiles)} profiles...")

        real_profiles = []
        generated_profiles = []
        individual_results = []

        for i, profile in enumerate(test_profiles):
            print(f"\nProfile {i + 1}/{len(test_profiles)}")

            # Extract real profile
            bean_temp_data = profile['roast_profile']['bean_temp']
            real_temps = np.array([p['value'] for p in bean_temp_data])

            # TODO: Extract conditioning from metadata
            # For now, using placeholder
            conditioning = torch.zeros(1, 256, device=self.device)  # Placeholder

            # Generate profile
            gen_temps = self.generate_profile(
                conditioning,
                start_temp=real_temps[0],
                num_steps=len(real_temps)
            )

            # Evaluate
            metrics = comprehensive_evaluation(real_temps, gen_temps, verbose=True)

            # Save
            real_profiles.append(real_temps)
            generated_profiles.append(gen_temps)
            individual_results.append(metrics)

            # Save comparison plot
            product_name = profile['metadata'].get('product_name', f'profile_{i}')
            plot_path = output_path / f"comparison_{product_name}.png"
            plot_comparison(
                real_temps, gen_temps,
                title=f"{product_name} - Real vs Generated",
                save_path=str(plot_path)
            )

        # Aggregate results
        batch_results = evaluate_batch(real_profiles, generated_profiles, verbose=True)

        # Check success criteria
        success_criteria = compute_success_metrics(batch_results['aggregate'])

        print("\n" + "=" * 80)
        print("SUCCESS CRITERIA")
        print("=" * 80)
        for criterion, passed in success_criteria.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {criterion}: {passed}")
        print("=" * 80)

        # Save results
        results = {
            'num_profiles': len(test_profiles),
            'aggregate_metrics': batch_results['aggregate'],
            'individual_results': individual_results,
            'success_criteria': success_criteria,
            'config': self.config,
            'epoch': self.epoch
        }

        results_path = output_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj

            json.dump(results, f, indent=2, default=convert)

        print(f"\n✓ Results saved to {output_path}/")

        return results


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate RoastFormer")
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', required=True,
                       help='Dataset directory for evaluation')
    parser.add_argument('--output', default='results/evaluation',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int,
                       help='Limit number of samples to evaluate')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ROASTFORMER EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")

    # TODO: Load test data and initialize model
    # This is a placeholder structure

    print("\n⚠️  Evaluation pipeline structure is ready!")
    print("⚠️  Next step: Integrate with actual data loading")


if __name__ == "__main__":
    main()
