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
        Autoregressive generation of roast profile

        Args:
            features: Dict of features (batch_size=1)
            start_temp: Initial charge temperature (°F)
            target_duration: Target profile length in seconds
            device: torch device

        Returns:
            generated_temps: (target_duration,) numpy array
        """
        self.eval()

        # Initialize with start temperature
        generated = torch.tensor([[start_temp]], device=device)  # (1, 1)

        with torch.no_grad():
            for t in range(1, target_duration):
                # Forward pass
                output = self.forward(generated, features)  # (1, t, 1)

                # Get next temperature prediction
                next_temp = output[0, -1, 0]  # Last predicted temp

                # Clamp to reasonable range
                next_temp = torch.clamp(next_temp, min=250.0, max=450.0)

                # Append to sequence
                next_temp = next_temp.unsqueeze(0).unsqueeze(0)  # (1, 1)
                generated = torch.cat([generated, next_temp], dim=1)  # (1, t+1)

        # Convert to numpy
        generated_temps = generated[0].cpu().numpy()

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
