import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class AudioTemporalTransformer(nn.Module):
    def __init__(self, input_dim=2048, num_classes=6, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.1):
        """
        Temporal Transformer for sequence modeling of audio embeddings.
        - input_dim: 2048 (from PANNs CNN14)
        - num_classes: Number of violence categories
        """
        super(AudioTemporalTransformer, self).__init__()
        
        # Linear projection to reduce dimensionality if needed
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding for temporal order
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, 2048]
        seq_len might be 10 (for 10-second clips with 1-second features)
        """
        # 1. Project to Transformer dimension
        x = self.projection(x)
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. Apply Transformer Self-Attention
        # Output x: [batch_size, seq_len, hidden_dim]
        x = self.transformer_encoder(x)
        
        # 4. Global Average Pooling over the temporal dimension
        x = torch.mean(x, dim=1)
        
        # 5. Final Classification
        logits = self.fc_out(x)
        return logits

if __name__ == "__main__":
    # Test with dummy data
    # 2 audio clips, 10 seconds each, 2048 feature dim
    model = AudioTemporalTransformer()
    dummy_input = torch.randn(2, 10, 2048) 
    out = model(dummy_input)
    print(f"Output logits shape: {out.shape}") # Should be [2, 6] for 6 classes
