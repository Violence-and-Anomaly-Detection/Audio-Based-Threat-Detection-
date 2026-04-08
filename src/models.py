import torch
import torch.nn as nn
from pann_inference import AudioTagging

class AudioThreatDetectionModel(nn.Module):
    """
    Proposed Architecture:
    1. Pretrained PANNs (CNN14) Feature Extractor
    2. Temporal Transformer Encoder
    3. Classification Head (Violence vs. Normal)
    """
    def __init__(self, num_classes=1, device='cpu'):
        super(AudioThreatDetectionModel, self).__init__()
        
        # 1. PANNs Backbone
        # We don't need to load weights manually; pann_inference handles it
        self.backbone = AudioTagging(checkpoint_path=None, device=device)
        
        # 2. Transformer for Temporal Context (Sequence analysis)
        encoder_layers = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        # 3. Final Head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, Samples]
        with torch.no_grad():
            # Get 2048-dim features from PANNs
            features = self.backbone.get_intermediate_features(x) 
            
        # Reshape for transformer: [Batch, SeqLen=1, Dim=2048]
        features = features.unsqueeze(1)
        
        # Temporal analysis
        x = self.transformer(features)
        
        # Final prediction
        x = x.mean(dim=1)
        return self.classifier(x)
