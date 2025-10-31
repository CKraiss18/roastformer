"""
Visualization Tools for Roast Profiles

Functions for plotting and visualizing coffee roast profiles:
- Single profile plots (temperature + RoR)
- Comparison plots (real vs generated)
- Batch visualization (multiple profiles)
- Training progress plots (loss curves)

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
from pathlib import Path


def plot_roast_profile(
    temps: np.ndarray,
    times: Optional[np.ndarray] = None,
    title: str = "Roast Profile",
    save_path: Optional[str] = None,
    show_ror: bool = True,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot a single roast profile with temperature and Rate of Rise

    Args:
        temps: Temperature sequence (°F)
        times: Optional time sequence (seconds), defaults to 0, 1, 2, ...
        title: Plot title
        save_path: Optional path to save figure
        show_ror: If True, show Rate of Rise on secondary axis
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if times is None:
        times = np.arange(len(temps))

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot temperature
    color = 'tab:red'
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Temperature (°F)', color=color, fontsize=12)
    ax1.plot(times, temps, color=color, linewidth=2, label='Bean Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Mark key points
    turning_point_idx = np.argmin(temps)
    ax1.scatter([times[turning_point_idx]], [temps[turning_point_idx]],
               color='blue', s=100, zorder=5, label=f'Turning Point ({temps[turning_point_idx]:.1f}°F)')

    # Mark first crack if it occurs
    first_crack_idx = np.argmax(temps >= 385) if np.any(temps >= 385) else None
    if first_crack_idx is not None and first_crack_idx > 0:
        ax1.axhline(y=385, color='green', linestyle='--', alpha=0.5, label='First Crack (~385°F)')
        ax1.scatter([times[first_crack_idx]], [temps[first_crack_idx]],
                   color='green', s=100, zorder=5)

    ax1.legend(loc='upper left')

    # Plot Rate of Rise if requested
    if show_ror and len(temps) > 1:
        ax2 = ax1.twinx()
        ror = np.diff(temps) / np.diff(times) * 60  # °F/min
        ror_times = times[1:]

        color = 'tab:blue'
        ax2.set_ylabel('Rate of Rise (°F/min)', color=color, fontsize=12)
        ax2.plot(ror_times, ror, color=color, linewidth=1.5, alpha=0.7, label='RoR')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    # Title and layout
    plt.title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")

    return fig


def plot_comparison(
    real: np.ndarray,
    generated: np.ndarray,
    real_times: Optional[np.ndarray] = None,
    gen_times: Optional[np.ndarray] = None,
    title: str = "Real vs Generated Profile",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot real vs generated roast profiles for comparison

    Args:
        real: Real temperature profile
        generated: Generated temperature profile
        real_times: Optional times for real profile
        gen_times: Optional times for generated profile
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if real_times is None:
        real_times = np.arange(len(real))
    if gen_times is None:
        gen_times = np.arange(len(generated))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # Plot temperatures
    ax1.plot(real_times, real, color='blue', linewidth=2, label='Real', alpha=0.8)
    ax1.plot(gen_times, generated, color='red', linewidth=2, label='Generated', alpha=0.8, linestyle='--')
    ax1.set_ylabel('Temperature (°F)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot error (if same length)
    min_len = min(len(real), len(generated))
    if min_len > 0:
        error = real[:min_len] - generated[:min_len]
        ax2.plot(real_times[:min_len], error, color='purple', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax2.fill_between(real_times[:min_len], error, 0, alpha=0.3, color='purple')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Error (°F)', fontsize=12)
        ax2.set_title(f'Difference (MAE: {np.mean(np.abs(error)):.2f}°F)', fontsize=12)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")

    return fig


def plot_batch_profiles(
    profiles: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Batch Roast Profiles",
    save_path: Optional[str] = None,
    max_profiles: int = 10,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot multiple roast profiles on the same axes

    Args:
        profiles: List of temperature profiles
        labels: Optional labels for each profile
        title: Plot title
        save_path: Optional path to save figure
        max_profiles: Maximum number of profiles to plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if labels is None:
        labels = [f"Profile {i+1}" for i in range(len(profiles))]

    # Limit number of profiles for readability
    profiles = profiles[:max_profiles]
    labels = labels[:max_profiles]

    fig, ax = plt.subplots(figsize=figsize)

    # Use a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    for profile, label, color in zip(profiles, labels, colors):
        times = np.arange(len(profile))
        ax.plot(times, profile, linewidth=2, label=label, alpha=0.7, color=color)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Temperature (°F)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved batch plot to {save_path}")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot training and validation loss curves

    Args:
        train_losses: Training loss per epoch
        val_losses: Optional validation loss per epoch
        metrics: Optional dict of metric_name -> values per epoch
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    num_plots = 1 if metrics is None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    if num_plots == 1:
        axes = [axes]

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, linewidth=2, label='Train Loss', color='blue')

    if val_losses is not None:
        axes[0].plot(epochs, val_losses, linewidth=2, label='Val Loss', color='red')

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot metrics if provided
    if metrics is not None:
        for metric_name, values in metrics.items():
            axes[1].plot(epochs[:len(values)], values, linewidth=2, label=metric_name)

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Metric Value', fontsize=12)
        axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")

    return fig


def save_profile_visualization(
    profile_data: Dict,
    output_dir: str,
    prefix: str = "profile"
):
    """
    Save a comprehensive visualization of a profile

    Args:
        profile_data: Dict with 'roast_profile' and 'metadata'
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    bean_temp_data = profile_data['roast_profile']['bean_temp']
    temps = np.array([point['value'] for point in bean_temp_data])
    times = np.array([point['time'] for point in bean_temp_data])

    # Get metadata
    metadata = profile_data.get('metadata', {})
    product_name = metadata.get('product_name', 'Unknown')
    origin = metadata.get('origin', 'Unknown')
    roast_level = metadata.get('roast_level', 'Unknown')

    title = f"{product_name}\n{origin} | {roast_level}"

    # Create plot
    save_path = output_path / f"{prefix}_{product_name}.png"
    plot_roast_profile(
        temps, times,
        title=title,
        save_path=str(save_path),
        show_ror=True
    )

    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Testing visualization module...")

    # Create synthetic profile
    temps = np.concatenate([
        np.linspace(425, 300, 100),  # Initial drop
        np.linspace(300, 405, 500)   # Heat up
    ])

    # Plot it
    fig = plot_roast_profile(
        temps,
        title="Test Roast Profile",
        show_ror=True
    )

    plt.show()
    print("✓ Visualization module working correctly!")
