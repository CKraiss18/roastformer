"""
Share Training Results with Claude

Creates a formatted summary of training results that you can easily
paste into Claude for analysis and discussion.

This helps maintain continuity across sessions by providing Claude
with all the key information about your training runs.

Author: Charlee Kraiss
Date: November 2024
"""

import json
import os
from pathlib import Path
from datetime import datetime
import argparse


def load_checkpoint_info(checkpoint_path):
    """Load information from a checkpoint file"""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    return {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
        'config': checkpoint.get('config', {}),
        'feature_dims': checkpoint.get('feature_dims', {}),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', [])
    }


def create_results_summary(results_json=None, checkpoint_path=None):
    """Create a formatted summary for Claude"""

    print("="*80)
    print("CREATING RESULTS SUMMARY FOR CLAUDE")
    print("="*80)

    summary_parts = []

    # Header
    summary_parts.append("# 🤖 RoastFormer Training Results")
    summary_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_parts.append("")

    # Load results
    results = None
    if results_json and os.path.exists(results_json):
        with open(results_json, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded: {results_json}")
    elif checkpoint_path and os.path.exists(checkpoint_path):
        results = load_checkpoint_info(checkpoint_path)
        print(f"✅ Loaded: {checkpoint_path}")
    else:
        # Try default locations
        default_results = 'results/transformer_training_results.json'
        default_checkpoint = 'checkpoints/best_transformer_model.pt'

        if os.path.exists(default_results):
            with open(default_results, 'r') as f:
                results = json.load(f)
            print(f"✅ Loaded: {default_results}")
        elif os.path.exists(default_checkpoint):
            results = load_checkpoint_info(default_checkpoint)
            print(f"✅ Loaded: {default_checkpoint}")
        else:
            print("❌ No results found!")
            print("   Please provide --results or --checkpoint")
            return None

    # Model Configuration
    summary_parts.append("## 🏗️ Model Configuration")
    summary_parts.append("")
    config = results.get('config', {})

    summary_parts.append(f"- **Architecture:** Decoder-only Transformer")
    summary_parts.append(f"- **d_model:** {config.get('d_model', 'N/A')}")
    summary_parts.append(f"- **Layers:** {config.get('num_layers', 'N/A')}")
    summary_parts.append(f"- **Attention heads:** {config.get('nhead', 'N/A')}")
    summary_parts.append(f"- **FFN dimension:** {config.get('dim_feedforward', 'N/A')}")
    summary_parts.append(f"- **Positional encoding:** {config.get('positional_encoding', 'N/A')}")
    summary_parts.append(f"- **Dropout:** {config.get('dropout', 'N/A')}")
    summary_parts.append(f"- **Parameters:** {results.get('num_parameters', 'N/A'):,}" if isinstance(results.get('num_parameters'), int) else f"- **Parameters:** {results.get('num_parameters', 'N/A')}")
    summary_parts.append("")

    # Training Configuration
    summary_parts.append("## 📚 Training Configuration")
    summary_parts.append("")
    summary_parts.append(f"- **Batch size:** {config.get('batch_size', 'N/A')}")
    summary_parts.append(f"- **Learning rate:** {config.get('learning_rate', 'N/A')}")
    summary_parts.append(f"- **Weight decay:** {config.get('weight_decay', 'N/A')}")
    summary_parts.append(f"- **Max sequence length:** {config.get('max_sequence_length', 'N/A')}")
    summary_parts.append(f"- **Epochs planned:** {config.get('num_epochs', 'N/A')}")
    summary_parts.append(f"- **Early stopping patience:** {config.get('early_stopping_patience', 'None')}")
    summary_parts.append("")

    # Dataset Information
    summary_parts.append("## 📊 Dataset")
    summary_parts.append("")
    feature_dims = results.get('feature_dims', {})
    summary_parts.append(f"- **Origins:** {feature_dims.get('num_origins', 'N/A')}")
    summary_parts.append(f"- **Processes:** {feature_dims.get('num_processes', 'N/A')}")
    summary_parts.append(f"- **Roast levels:** {feature_dims.get('num_roast_levels', 'N/A')}")
    summary_parts.append(f"- **Varieties:** {feature_dims.get('num_varieties', 'N/A')}")
    summary_parts.append(f"- **Unique flavors:** {feature_dims.get('num_flavors', 'N/A')}")
    summary_parts.append("")

    # Training Results
    summary_parts.append("## 📈 Training Results")
    summary_parts.append("")

    final_epoch = results.get('final_epoch', results.get('epoch', 'N/A'))
    best_val = results.get('best_val_loss', 'N/A')
    train_losses = results.get('train_losses', [])
    val_losses = results.get('val_losses', [])

    summary_parts.append(f"- **Epochs completed:** {final_epoch}")
    summary_parts.append(f"- **Best validation loss:** {best_val:.4f}°F" if isinstance(best_val, (int, float)) else f"- **Best validation loss:** {best_val}")

    if train_losses and val_losses:
        summary_parts.append(f"- **Final train loss:** {train_losses[-1]:.4f}°F")
        summary_parts.append(f"- **Final val loss:** {val_losses[-1]:.4f}°F")

        # Calculate improvement
        if len(train_losses) > 1:
            train_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
            val_improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0]) * 100
            summary_parts.append(f"- **Train loss improvement:** {train_improvement:.1f}%")
            summary_parts.append(f"- **Val loss improvement:** {val_improvement:.1f}%")

        # Check for overfitting
        if train_losses[-1] < val_losses[-1] * 0.5:
            summary_parts.append(f"- ⚠️ **Overfitting detected:** Val loss much higher than train loss")

    summary_parts.append("")

    # Loss History (last 10 epochs)
    if train_losses and val_losses and len(train_losses) > 0:
        summary_parts.append("## 📉 Recent Loss History (last 10 epochs)")
        summary_parts.append("")
        summary_parts.append("| Epoch | Train Loss | Val Loss |")
        summary_parts.append("|-------|-----------|----------|")

        start_idx = max(0, len(train_losses) - 10)
        for i in range(start_idx, len(train_losses)):
            epoch_num = i + 1
            summary_parts.append(f"| {epoch_num} | {train_losses[i]:.4f}°F | {val_losses[i]:.4f}°F |")

        summary_parts.append("")

    # Success Metrics Analysis
    summary_parts.append("## 🎯 Success Metrics (Target from CLAUDE.md)")
    summary_parts.append("")
    summary_parts.append("| Metric | Target | Current | Status |")
    summary_parts.append("|--------|--------|---------|--------|")

    # Temperature MAE
    mae_target = 5.0
    current_mae = best_val if isinstance(best_val, (int, float)) else None
    if current_mae:
        mae_status = "✅" if current_mae < mae_target else "⚠️"
        summary_parts.append(f"| Temperature MAE | < {mae_target}°F | {current_mae:.4f}°F | {mae_status} |")
    else:
        summary_parts.append(f"| Temperature MAE | < {mae_target}°F | N/A | ⏳ |")

    summary_parts.append(f"| Finish Temp Accuracy | >90% (±10°F) | *Need evaluation* | ⏳ |")
    summary_parts.append(f"| Monotonicity | 100% | *Need evaluation* | ⏳ |")
    summary_parts.append(f"| Bounded RoR | >95% | *Need evaluation* | ⏳ |")

    summary_parts.append("")

    # Next Steps
    summary_parts.append("## 🚀 Next Steps")
    summary_parts.append("")
    summary_parts.append("1. **Evaluate model on validation set:**")
    summary_parts.append("   ```bash")
    summary_parts.append("   python evaluate_transformer.py --plot --num_samples 10")
    summary_parts.append("   ```")
    summary_parts.append("")
    summary_parts.append("2. **Generate sample profiles:**")
    summary_parts.append("   ```bash")
    summary_parts.append("   python generate_profiles.py --origin Ethiopia --flavors 'berries,floral' --plot")
    summary_parts.append("   ```")
    summary_parts.append("")
    summary_parts.append("3. **Run ablation studies:**")
    summary_parts.append("   - Try different positional encodings")
    summary_parts.append("   - Test different model sizes")
    summary_parts.append("   - Compare configurations")
    summary_parts.append("")

    # Questions for Claude
    summary_parts.append("## 💬 Questions for Claude")
    summary_parts.append("")
    summary_parts.append("- Is the validation loss acceptable?")
    summary_parts.append("- Any signs of overfitting or underfitting?")
    summary_parts.append("- Should I adjust hyperparameters?")
    summary_parts.append("- Ready for ablation studies?")
    summary_parts.append("")

    # Combine all parts
    summary_text = "\n".join(summary_parts)

    # Save to file
    output_file = f"results/claude_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs('results', exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(summary_text)

    print(f"✅ Summary saved to: {output_file}")
    print("")
    print("="*80)
    print("COPY AND PASTE THIS TO CLAUDE")
    print("="*80)
    print("")
    print(summary_text)
    print("")
    print("="*80)

    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description='Create a summary of training results to share with Claude'
    )

    parser.add_argument(
        '--results',
        type=str,
        default=None,
        help='Path to transformer_training_results.json'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint .pt file'
    )

    args = parser.parse_args()

    summary = create_results_summary(
        results_json=args.results,
        checkpoint_path=args.checkpoint
    )

    if summary:
        print("\n✅ Summary created! Copy the text above and paste it in your conversation with Claude.")
        print("   This helps Claude understand your training results and provide better guidance.")
    else:
        print("\n❌ Could not create summary. Please check that you have training results.")


if __name__ == "__main__":
    main()
