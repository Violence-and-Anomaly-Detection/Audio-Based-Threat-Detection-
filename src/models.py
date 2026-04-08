import torch
import torch.nn as nn
import torch.nn.functional as F
from pann_inference import AudioTagging

class AdvancedThreatModel(nn.Module):
    """
    Module 2: Multimodal Violence Detection
    Backbone: PANNs (CNN14)
    Temporal: Transformer Encoder
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(AdvancedThreatModel, self).__init__()
        
        # Load PANNs (Pretrained Audio Neural Network)
        # We use CNN14 as the backbone for high-accuracy feature extraction
        self.backbone = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transformer Layer for Temporal Sequence Modeling
        # This handles the "Temporal Models" part of your project title
        encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Binary Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. Extract intermediate features from PANNs
        # PANNs returns features of size [Batch, 2048] for each audio segment
        with torch.no_grad():
            features = self.backbone.get_intermediate_features(x)
        
        # 2. Add Temporal Context via Transformer
        # (Assuming segments are treated as a sequence)
        x = self.transformer(features.unsqueeze(1)) # Adding temporal dimension
        
        # 3. Global Decision
        x = x.mean(dim=1)
        return self.fc(x)

class SimpleBaselineCNN(nn.Module):
    """
    A simpler version of your morning model if you want a fast baseline.
    """
    def __init__(self, num_classes=1):
        super(SimpleBaselineCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
