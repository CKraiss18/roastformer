"""
Adapter Module for RoastFormer Transformer

Bridges the gap between preprocessed data format and RoastFormer architecture.
Adapts the ConditioningModule to work with the PreprocessedDataLoader output.

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import torch
import torch.nn as nn
from typing import Dict


class AdaptedConditioningModule(nn.Module):
    """
    Conditioning module adapted for preprocessed data format

    Takes the output from PreprocessedDataLoader and creates
    conditioning vectors for the transformer.
    """

    def __init__(
        self,
        num_origins: int,
        num_processes: int,
        num_roast_levels: int,
        num_varieties: int,
        num_flavors: int,
        embed_dim: int = 32
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Categorical embeddings
        self.origin_embed = nn.Embedding(num_origins, embed_dim)
        self.process_embed = nn.Embedding(num_processes, embed_dim)
        self.roast_level_embed = nn.Embedding(num_roast_levels, embed_dim)
        self.variety_embed = nn.Embedding(num_varieties, embed_dim)

        # Flavor projection (from multi-hot vector to embedding)
        self.flavor_proj = nn.Linear(num_flavors, embed_dim)

        # Continuous feature projection (3 features: target_temp, altitude, density)
        self.continuous_proj = nn.Linear(3, embed_dim)

        # Total conditioning dimension: 6 embeddings * embed_dim
        # (origin, process, roast_level, variety, flavor, continuous)
        self.condition_dim = embed_dim * 6

    def forward(self, features: Dict) -> torch.Tensor:
        """
        Create conditioning vector from preprocessed features

        Args:
            features: Dict from PreprocessedDataLoader with structure:
                {
                    'categorical': {
                        'origin': (batch, 1),
                        'process': (batch, 1),
                        'roast_level': (batch, 1),
                        'variety': (batch, 1)
                    },
                    'continuous': {
                        'target_finish_temp': (batch, 1),
                        'altitude': (batch, 1),
                        'bean_density': (batch, 1)
                    },
                    'flavors': (batch, num_flavors)
                }

        Returns:
            condition_vector: (batch, condition_dim)
        """
        batch_size = features['categorical']['origin'].shape[0]

        # Get categorical embeddings (squeeze to remove extra dimension)
        origin_emb = self.origin_embed(features['categorical']['origin'].squeeze(-1))  # (batch, embed)
        process_emb = self.process_embed(features['categorical']['process'].squeeze(-1))
        roast_level_emb = self.roast_level_embed(features['categorical']['roast_level'].squeeze(-1))
        variety_emb = self.variety_embed(features['categorical']['variety'].squeeze(-1))

        # Project flavor multi-hot vector
        flavor_emb = self.flavor_proj(features['flavors'])  # (batch, embed)

        # Combine continuous features and project
        continuous_features = torch.cat([
            features['continuous']['target_finish_temp'],
            features['continuous']['altitude'],
            features['continuous']['bean_density']
        ], dim=1)  # (batch, 3)

        continuous_emb = self.continuous_proj(continuous_features)  # (batch, embed)

        # Concatenate all embeddings
        condition_vector = torch.cat([
            origin_emb,
            process_emb,
            roast_level_emb,
            variety_emb,
            flavor_emb,
            continuous_emb
        ], dim=1)  # (batch, 6 * embed)

        return condition_vector


class AdaptedRoastFormer(nn.Module):
    """
    RoastFormer adapted to work with PreprocessedDataLoader

    This wraps the full transformer architecture with an adapted
    conditioning module that handles the preprocessed data format.
    """

    def __init__(
        self,
        conditioning_module: AdaptedConditioningModule,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        positional_encoding: str = 'sinusoidal',
        max_seq_len: int = 1000
    ):
        super().__init__()

        self.conditioning_module = conditioning_module
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Project conditioning vector to model dimension
        self.condition_proj = nn.Linear(conditioning_module.condition_dim, d_model)

        # Input embedding for temperature values
        self.temp_embed = nn.Linear(1, d_model)

        # Positional encoding
        if positional_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        elif positional_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len)
        elif positional_encoding == 'rope':
            self.pos_encoding = RoPEPositionalEncoding(d_model, max_seq_len)
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection to temperature
        self.output_proj = nn.Linear(d_model, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, temps: torch.Tensor, features: Dict, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for training

        Args:
            temps: (batch, seq_len) - Temperature sequence
            features: Dict of features from PreprocessedDataLoader
            mask: (batch, seq_len) - Boolean mask for valid timesteps

        Returns:
            predicted_temps: (batch, seq_len, 1)
        """
        batch_size, seq_len = temps.shape

        # Get conditioning vector
        condition_vector = self.conditioning_module(features)  # (batch, condition_dim)

        # Project condition to model dimension
        condition_embed = self.condition_proj(condition_vector)  # (batch, d_model)
        condition_embed = condition_embed.unsqueeze(1)  # (batch, 1, d_model)

        # Prepare condition as memory for cross-attention
        condition_memory = condition_embed  # (batch, 1, d_model)

        # Embed temperature sequence
        temps_input = temps.unsqueeze(-1)  # (batch, seq_len, 1)
        temp_embed = self.temp_embed(temps_input)  # (batch, seq_len, d_model)

        # Add positional encoding
        temp_embed = self.pos_encoding(temp_embed)

        # Add condition to input (broadcast across sequence)
        temp_embed = temp_embed + condition_embed

        # Generate causal mask if not provided
        if mask is None:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(temps.device)
        else:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(temps.device)

        # Transformer decoder with cross-attention to conditioning
        output = self.transformer(
            tgt=temp_embed,
            memory=condition_memory,
            tgt_mask=causal_mask
        )

        # Layer norm
        output = self.layer_norm(output)

        # Project to temperature
        predicted_temps = self.output_proj(output)  # (batch, seq_len, 1)

        return predicted_temps

    def generate(
        self,
        features: Dict,
        start_temp: float = 426.0,
        target_duration: int = 600,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Autoregressive generation of roast profile (WITH NORMALIZATION)

        Args:
            features: Dict of features (batch_size=1)
            start_temp: Initial charge temperature in °F (e.g., 426.0)
            target_duration: Target profile length in seconds
            device: torch device

        Returns:
            generated_temps: (target_duration,) numpy array in °F (denormalized)
        """
        from src.dataset.preprocessed_data_loader_NORMALIZED import normalize_temperature, denormalize_temperature
        import numpy as np

        self.eval()

        # Normalize start temperature to [0, 1] range
        start_temp_norm = normalize_temperature(start_temp)
        generated = torch.tensor([[start_temp_norm]], device=device)  # (1, 1) - NORMALIZED

        with torch.no_grad():
            for t in range(1, target_duration):
                # Forward pass (model trained on normalized temps)
                output = self.forward(generated, features)  # (1, t, 1)

                # Get next temperature prediction (in normalized space [0, 1])
                next_temp_norm = output[0, -1, 0]  # Last predicted temp

                # Clamp to [0, 1] range
                next_temp_norm = torch.clamp(next_temp_norm, min=0.0, max=1.0)

                # Append to sequence (stay in normalized space)
                next_temp_norm = next_temp_norm.unsqueeze(0).unsqueeze(0)  # (1, 1)
                generated = torch.cat([generated, next_temp_norm], dim=1)  # (1, t+1)

        # Convert to numpy (still normalized)
        generated_norm = generated[0].cpu().numpy()

        # Denormalize to °F for output
        generated_temps = np.array([denormalize_temperature(t) for t in generated_norm])

        return generated_temps


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        return x


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from Su et al. (2021)
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Key advantages for time-series:
    - Encodes relative position information through rotation
    - No fixed maximum sequence length
    - Better long-range dependency modeling
    - Position-aware attention (distance matters)

    Implementation: Applies rotation matrices to input based on position,
    encoding both absolute and relative positional information.
    """

    def __init__(self, d_model: int, max_len: int = 2000, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Compute inverse frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        # These determine the rotation frequency for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute rotation matrices for efficiency
        self._precompute_rotations(max_len)

    def _precompute_rotations(self, seq_len: int):
        """Precompute cos and sin for all positions up to seq_len"""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Compute angles: outer product of positions and inverse frequencies
        # freqs shape: (seq_len, d_model/2)
        freqs = torch.outer(t, self.inv_freq)

        # Compute cos and sin for rotation matrices
        # These will be used to rotate the embeddings
        cos = freqs.cos()  # (seq_len, d_model/2)
        sin = freqs.sin()  # (seq_len, d_model/2)

        # Expand to full d_model by repeating each value
        # This handles the pair-wise rotation structure of RoPE
        cos = torch.cat([cos, cos], dim=-1)  # (seq_len, d_model)
        sin = torch.cat([sin, sin], dim=-1)  # (seq_len, d_model)

        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the dimensions of x.

        For x = [x1, x2, x3, x4, ...], returns [-x2, x1, -x4, x3, ...]

        This implements the core rotation operation:
        - Split dimensions into pairs
        - Rotate each pair by 90 degrees

        This is the key insight of RoPE: rotation in 2D subspaces
        encodes position information in a way that captures relative distances.
        """
        # Split into first half and second half of dimensions
        x1 = x[..., : x.shape[-1] // 2]  # First half
        x2 = x[..., x.shape[-1] // 2 :]  # Second half

        # Rotation: swap and negate second half
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.

        The rotation formula:
            RoPE(x, pos) = x * cos(pos * θ) + rotate_half(x) * sin(pos * θ)

        where θ are the inverse frequencies, and pos is the position.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            seq_len: Sequence length

        Returns:
            Tensor with rotary position encoding applied
        """
        # Extend cache if sequence is longer than precomputed
        if seq_len > self.max_len:
            self._precompute_rotations(seq_len)
            self.max_len = seq_len

        # Get cos and sin for this sequence length
        cos = self.cos_cached[:seq_len, :]  # (seq_len, d_model)
        sin = self.sin_cached[:seq_len, :]  # (seq_len, d_model)

        # Expand to batch dimension: (1, seq_len, d_model)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Apply rotation: x * cos + rotate_half(x) * sin
        # This encodes position through rotation angle = position * frequency
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input embeddings.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Tensor with rotary position encoding applied, same shape as input

        Note: Unlike additive positional encodings (sinusoidal, learned),
        RoPE modifies the representations through rotation rather than addition.
        This preserves the magnitude of embeddings while encoding position.
        """
        seq_len = x.shape[1]
        return self._apply_rotary_emb(x, seq_len)
