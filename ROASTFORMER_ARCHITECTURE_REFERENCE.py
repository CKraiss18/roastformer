"""
RoastFormer Transformer Architecture Reference
Complete code examples for conditioning, embeddings, and model architecture
Consolidated from all previous discussions
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# SECTION 1: FEATURE ENCODING & CONDITIONING
# =============================================================================

class FeatureEncoder:
    """
    Encode all Phase 1 + Phase 2 + Flavor features for transformer conditioning
    """
    
    def __init__(self, df):
        """Initialize encoders from dataset"""
        
        # Categorical encoders
        self.origin_encoder = LabelEncoder()
        self.process_encoder = LabelEncoder()
        self.variety_encoder = LabelEncoder()
        self.roast_level_encoder = LabelEncoder()
        
        # Fit on unique values
        self.origin_encoder.fit(df['origin'].dropna().unique())
        self.process_encoder.fit(df['process'].dropna().unique())
        self.variety_encoder.fit(df['variety'].dropna().unique())
        self.roast_level_encoder.fit(df['roast_level'].dropna().unique())
        
        # Store vocabulary sizes
        self.num_origins = len(self.origin_encoder.classes_)
        self.num_processes = len(self.process_encoder.classes_)
        self.num_varieties = len(self.variety_encoder.classes_)
        self.num_roast_levels = len(self.roast_level_encoder.classes_)
        
        # Flavor vocabulary (built from dataset)
        self.flavor_vocab = self._build_flavor_vocabulary(df)
        self.num_flavors = len(self.flavor_vocab)
        
        print(f"Feature Encoder Initialized:")
        print(f"  Origins: {self.num_origins}")
        print(f"  Processes: {self.num_processes}")
        print(f"  Varieties: {self.num_varieties}")
        print(f"  Roast Levels: {self.num_roast_levels}")
        print(f"  Flavor Vocabulary: {self.num_flavors}")
    
    def _build_flavor_vocabulary(self, df):
        """Build flavor vocabulary from all products"""
        all_flavors = set()
        
        # Extract all unique flavor notes from dataset
        for notes in df['flavor_notes_parsed'].dropna():
            if isinstance(notes, list):
                all_flavors.update([f.lower() for f in notes])
        
        # Common flavor categories
        common_flavors = [
            'berries', 'cherry', 'citrus', 'stone fruit', 'tropical',
            'chocolate', 'cocoa', 'caramel', 'toffee', 'brown sugar',
            'floral', 'honeysuckle', 'jasmine', 'rose',
            'nutty', 'almond', 'hazelnut',
            'tea', 'earl grey', 'black tea',
            'spice', 'cinnamon', 'vanilla',
            'round', 'creamy', 'smooth', 'silky', 'juicy'
        ]
        
        all_flavors.update(common_flavors)
        
        # Create flavor to index mapping
        flavor_vocab = {flavor: idx for idx, flavor in enumerate(sorted(all_flavors))}
        return flavor_vocab
    
    def encode_categorical(self, row):
        """Encode categorical features to indices"""
        
        origin_idx = self.origin_encoder.transform([row['origin']])[0] if pd.notna(row['origin']) else 0
        process_idx = self.process_encoder.transform([row['process']])[0] if pd.notna(row['process']) else 0
        variety_idx = self.variety_encoder.transform([row['variety']])[0] if pd.notna(row['variety']) else 0
        roast_idx = self.roast_level_encoder.transform([row['roast_level']])[0] if pd.notna(row['roast_level']) else 0
        
        return {
            'origin': origin_idx,
            'process': process_idx,
            'variety': variety_idx,
            'roast_level': roast_idx
        }
    
    def encode_continuous(self, row):
        """Encode continuous features (normalized 0-1)"""
        
        # Normalization ranges based on expected values
        finish_temp_norm = row['target_finish_temp'] / 425.0  # Max ~425°F
        altitude_norm = row['altitude_numeric'] / 2500.0 if pd.notna(row['altitude_numeric']) else 0.6
        density_norm = row['bean_density_proxy'] / 0.80 if pd.notna(row['bean_density_proxy']) else 0.85
        caffeine_norm = row['caffeine_mg'] / 230.0 if pd.notna(row['caffeine_mg']) else 0.9
        
        return torch.tensor([
            finish_temp_norm,
            altitude_norm,
            density_norm,
            caffeine_norm
        ], dtype=torch.float32)
    
    def encode_flavors_onehot(self, flavor_notes):
        """
        One-hot encoding of flavor notes
        
        Args:
            flavor_notes: List of flavor strings, e.g., ['berries', 'chocolate', 'floral']
        
        Returns:
            torch.Tensor of shape (num_flavors,) with 1s for present flavors
        """
        flavor_vector = torch.zeros(self.num_flavors)
        
        if flavor_notes:
            for flavor in flavor_notes:
                flavor_lower = flavor.lower()
                if flavor_lower in self.flavor_vocab:
                    idx = self.flavor_vocab[flavor_lower]
                    flavor_vector[idx] = 1.0
        
        return flavor_vector
    
    def encode_flavors_embedding(self, flavor_notes, flavor_embedding_layer):
        """
        Embedding-based encoding of flavor notes
        
        Args:
            flavor_notes: List of flavor strings
            flavor_embedding_layer: nn.Embedding layer
        
        Returns:
            Averaged flavor embeddings
        """
        if not flavor_notes:
            # Return zero embedding if no flavors
            return torch.zeros(flavor_embedding_layer.embedding_dim)
        
        flavor_indices = []
        for flavor in flavor_notes:
            flavor_lower = flavor.lower()
            if flavor_lower in self.flavor_vocab:
                flavor_indices.append(self.flavor_vocab[flavor_lower])
        
        if not flavor_indices:
            return torch.zeros(flavor_embedding_layer.embedding_dim)
        
        # Get embeddings and average
        flavor_embeds = flavor_embedding_layer(torch.tensor(flavor_indices))
        avg_flavor_embed = torch.mean(flavor_embeds, dim=0)
        
        return avg_flavor_embed


# =============================================================================
# SECTION 2: CONDITIONING MODULE
# =============================================================================

class ConditioningModule(nn.Module):
    """
    Combines all features into a unified conditioning vector
    Supports multiple conditioning strategies
    """
    
    def __init__(self, feature_encoder, embed_dim=32, use_flavor_embeddings=True):
        super().__init__()
        
        self.feature_encoder = feature_encoder
        self.embed_dim = embed_dim
        self.use_flavor_embeddings = use_flavor_embeddings
        
        # Categorical embeddings
        self.origin_embed = nn.Embedding(feature_encoder.num_origins, embed_dim)
        self.process_embed = nn.Embedding(feature_encoder.num_processes, embed_dim)
        self.variety_embed = nn.Embedding(feature_encoder.num_varieties, embed_dim)
        self.roast_level_embed = nn.Embedding(feature_encoder.num_roast_levels, embed_dim)
        
        # Flavor embeddings (if using embedding approach)
        if use_flavor_embeddings:
            self.flavor_embed = nn.Embedding(feature_encoder.num_flavors, embed_dim)
            self.flavor_proj = None
        else:
            # One-hot approach: project to embed_dim
            self.flavor_embed = None
            self.flavor_proj = nn.Linear(feature_encoder.num_flavors, embed_dim)
        
        # Continuous feature projection
        self.continuous_proj = nn.Linear(4, embed_dim)  # 4 continuous features
        
        # Total conditioning dimension
        if use_flavor_embeddings:
            self.condition_dim = embed_dim * 5  # origin, process, variety, roast, flavor
        else:
            self.condition_dim = embed_dim * 5  # same, but flavor is projected one-hot
        
        self.condition_dim += embed_dim  # Add continuous projection
    
    def forward(self, categorical_indices, continuous_features, flavor_notes=None):
        """
        Forward pass to create conditioning vector
        
        Args:
            categorical_indices: dict with keys ['origin', 'process', 'variety', 'roast_level']
            continuous_features: torch.Tensor of shape (4,) [finish_temp, altitude, density, caffeine]
            flavor_notes: list of flavor strings or None
        
        Returns:
            condition_vector: torch.Tensor of shape (condition_dim,)
        """
        # Get categorical embeddings
        origin_emb = self.origin_embed(torch.tensor(categorical_indices['origin']))
        process_emb = self.process_embed(torch.tensor(categorical_indices['process']))
        variety_emb = self.variety_embed(torch.tensor(categorical_indices['variety']))
        roast_emb = self.roast_level_embed(torch.tensor(categorical_indices['roast_level']))
        
        # Get flavor embedding/projection
        if self.use_flavor_embeddings:
            flavor_emb = self.feature_encoder.encode_flavors_embedding(flavor_notes, self.flavor_embed)
        else:
            flavor_onehot = self.feature_encoder.encode_flavors_onehot(flavor_notes)
            flavor_emb = self.flavor_proj(flavor_onehot)
        
        # Project continuous features
        continuous_emb = self.continuous_proj(continuous_features)
        
        # Concatenate all embeddings
        condition_vector = torch.cat([
            origin_emb,
            process_emb,
            variety_emb,
            roast_emb,
            flavor_emb,
            continuous_emb
        ])
        
        return condition_vector


# =============================================================================
# SECTION 3: POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pe(positions)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - used in modern LLMs
    More stable for long sequences
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create rotation angles
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * 
                            (-np.log(10000.0) / d_model))
        
        angles = position * div_term  # (seq_len, d_model/2)
        
        # Apply rotation to pairs of dimensions
        x_rotated = x.clone()
        x_rotated[:, :, 0::2] = x[:, :, 0::2] * torch.cos(angles) - x[:, :, 1::2] * torch.sin(angles)
        x_rotated[:, :, 1::2] = x[:, :, 0::2] * torch.sin(angles) + x[:, :, 1::2] * torch.cos(angles)
        
        return x_rotated


# =============================================================================
# SECTION 4: ROASTFORMER ARCHITECTURE
# =============================================================================

class RoastFormer(nn.Module):
    """
    Decoder-only Transformer for roast profile generation
    Conditioned on bean characteristics and target flavor profile
    """
    
    def __init__(
        self,
        conditioning_module,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        positional_encoding='sinusoidal',  # 'sinusoidal', 'learned', or 'rotary'
        max_seq_len=1000
    ):
        super().__init__()
        
        self.conditioning_module = conditioning_module
        self.d_model = d_model
        
        # Project conditioning vector to model dimension
        self.condition_proj = nn.Linear(conditioning_module.condition_dim, d_model)
        
        # Input embedding for temperature values
        self.temp_embed = nn.Linear(1, d_model)
        
        # Positional encoding
        if positional_encoding == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        elif positional_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len)
        elif positional_encoding == 'rotary':
            self.pos_encoding = RotaryPositionalEncoding(d_model)
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
    
    def forward(self, temps, categorical_indices, continuous_features, flavor_notes=None, mask=None):
        """
        Forward pass for training
        
        Args:
            temps: (batch, seq_len, 1) - Temperature sequence
            categorical_indices: dict of categorical feature indices
            continuous_features: (batch, 4) - Continuous features
            flavor_notes: list of flavor note lists (batch size)
            mask: Optional attention mask
        
        Returns:
            predicted_temps: (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = temps.shape
        
        # Get conditioning vector
        condition_vector = self.conditioning_module(
            categorical_indices, 
            continuous_features[0],  # Assume same condition for whole batch (simplification)
            flavor_notes[0] if flavor_notes else None
        )
        
        # Project condition to model dimension and expand to sequence
        condition_embed = self.condition_proj(condition_vector)  # (d_model,)
        condition_embed = condition_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
        condition_memory = condition_embed.expand(batch_size, 1, -1)  # (batch, 1, d_model)
        
        # Embed temperature sequence
        temp_embed = self.temp_embed(temps)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        temp_embed = self.pos_encoding(temp_embed)
        
        # Add condition to input (broadcast)
        temp_embed = temp_embed + condition_embed
        
        # Generate causal mask if not provided
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(temps.device)
        
        # Transformer decoder
        output = self.transformer(
            tgt=temp_embed,
            memory=condition_memory,
            tgt_mask=mask
        )
        
        # Layer norm
        output = self.layer_norm(output)
        
        # Project to temperature
        predicted_temps = self.output_proj(output)
        
        return predicted_temps
    
    def generate(self, categorical_indices, continuous_features, flavor_notes=None, 
                 start_temp=426.0, target_duration=600, device='cpu'):
        """
        Autoregressive generation of roast profile
        
        Args:
            categorical_indices: dict of feature indices
            continuous_features: (4,) tensor
            flavor_notes: list of flavor strings
            start_temp: Initial charge temperature (°F)
            target_duration: Target profile length in seconds
            device: torch device
        
        Returns:
            generated_temps: (target_duration,) numpy array
        """
        self.eval()
        
        # Initialize with start temperature
        generated = torch.tensor([[start_temp]], device=device).unsqueeze(0)  # (1, 1, 1)
        
        with torch.no_grad():
            for t in range(1, target_duration):
                # Forward pass
                output = self.forward(
                    generated,
                    categorical_indices,
                    continuous_features.unsqueeze(0),
                    [flavor_notes] if flavor_notes else None
                )
                
                # Get next temperature prediction
                next_temp = output[0, -1, 0]  # Last predicted temp
                
                # Append to sequence
                next_temp = next_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1)
                generated = torch.cat([generated, next_temp], dim=1)
        
        # Convert to numpy
        generated_temps = generated[0, :, 0].cpu().numpy()
        
        return generated_temps


# =============================================================================
# SECTION 5: TRAINING UTILITIES
# =============================================================================

class RoastProfileDataset(torch.utils.data.Dataset):
    """Dataset for roast profiles with conditioning"""
    
    def __init__(self, profiles_df, feature_encoder):
        self.df = profiles_df
        self.feature_encoder = feature_encoder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get categorical indices
        categorical = self.feature_encoder.encode_categorical(row)
        
        # Get continuous features
        continuous = self.feature_encoder.encode_continuous(row)
        
        # Get flavor notes
        flavor_notes = row['flavor_notes_parsed'] if 'flavor_notes_parsed' in row else None
        
        # Load temperature profile (from JSON)
        # This would load the actual profile data
        temps = self._load_temperature_profile(row)
        
        return {
            'temps': torch.tensor(temps, dtype=torch.float32).unsqueeze(-1),
            'categorical': categorical,
            'continuous': continuous,
            'flavor_notes': flavor_notes
        }
    
    def _load_temperature_profile(self, row):
        """Load temperature sequence from profile JSON"""
        # Load from JSON file
        import json
        profile_path = f"onyx_dataset/profiles/{row['product_name'].lower().replace(' ', '_')}.json"
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        temps = [p['value'] for p in profile['roast_profile']['bean_temp']]
        return temps


def train_roastformer(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cpu'):
    """
    Training loop for RoastFormer
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            temps = batch['temps'].to(device)
            categorical = batch['categorical']
            continuous = batch['continuous'].to(device)
            flavor_notes = batch['flavor_notes']
            
            # Teacher forcing: use all but last temp as input
            input_temps = temps[:, :-1, :]
            target_temps = temps[:, 1:, :]
            
            # Forward pass
            predictions = model(input_temps, categorical, continuous, flavor_notes)
            
            # Calculate loss
            loss = criterion(predictions, target_temps)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                temps = batch['temps'].to(device)
                categorical = batch['categorical']
                continuous = batch['continuous'].to(device)
                flavor_notes = batch['flavor_notes']
                
                input_temps = temps[:, :-1, :]
                target_temps = temps[:, 1:, :]
                
                predictions = model(input_temps, categorical, continuous, flavor_notes)
                loss = criterion(predictions, target_temps)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'roastformer_best.pth')
    
    return model


# =============================================================================
# SECTION 6: USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Complete example of using RoastFormer"""
    
    import pandas as pd
    
    # Load dataset
    df = pd.read_csv('onyx_dataset/dataset_summary.csv')
    
    # Initialize feature encoder
    feature_encoder = FeatureEncoder(df)
    
    # Create conditioning module
    conditioning_module = ConditioningModule(
        feature_encoder,
        embed_dim=32,
        use_flavor_embeddings=True
    )
    
    # Create RoastFormer model
    model = RoastFormer(
        conditioning_module=conditioning_module,
        d_model=256,
        nhead=8,
        num_layers=6,
        positional_encoding='sinusoidal'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example: Generate a roast profile
    categorical_indices = {
        'origin': 0,  # Ethiopia
        'process': 0,  # Washed
        'variety': 1,  # Heirloom
        'roast_level': 0  # Light
    }
    
    continuous_features = torch.tensor([
        395.0 / 425.0,  # Target finish temp (normalized)
        2000.0 / 2500.0,  # Altitude (normalized)
        0.75 / 0.80,  # Bean density (normalized)
        210.0 / 230.0  # Caffeine (normalized)
    ])
    
    flavor_notes = ['berries', 'floral', 'citrus']
    
    # Generate profile
    generated_profile = model.generate(
        categorical_indices=categorical_indices,
        continuous_features=continuous_features,
        flavor_notes=flavor_notes,
        start_temp=426.0,
        target_duration=600,
        device='cpu'
    )
    
    print(f"Generated profile shape: {generated_profile.shape}")
    print(f"Start temp: {generated_profile[0]:.1f}°F")
    print(f"Finish temp: {generated_profile[-1]:.1f}°F")
    
    return model, generated_profile


if __name__ == "__main__":
    # Run example
    model, profile = example_usage()
    
    # Visualize
    import matplotlib.pyplot as plt
    
    times = np.arange(len(profile))
    plt.figure(figsize=(12, 6))
    plt.plot(times, profile, linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (°F)')
    plt.title('Generated Roast Profile - Ethiopian Washed, Light Roast, Berries/Floral/Citrus')
    plt.grid(alpha=0.3)
    plt.show()
