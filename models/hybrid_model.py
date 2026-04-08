import torch
import torch.nn as nn
from models.panns_backbone import Cnn14
from models.transformer import AudioTemporalTransformer

class HybridAudioModel(nn.Module):
    def __init__(self, num_classes=6, freeze_backbone=True):
        """
        End-to-end Hybrid Model: PANNs Backbone + Temporal Transformer
        - freeze_backbone: If True, only train the Transformer (saves time/memory).
        """
        super(HybridAudioModel, self).__init__()
        
        # 1. Feature Extractor (Backbone)
        self.backbone = Cnn14()
        
        # 2. Temporal Aggregator (Transformer)
        self.temporal_model = AudioTemporalTransformer(
            input_dim=2048, 
            num_classes=num_classes
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: [batch_size, seq_len, 1, time_steps, mel_bins]
        Expects a sequence of spectrogram windows.
        """
        batch_size, seq_len, c, t, f = x.shape
        
        # 1. Reshape to feed all windows into backbone at once
        # [batch_size * seq_len, 1, time_steps, mel_bins]
        x = x.view(batch_size * seq_len, c, t, f)
        
        # 2. Extract 2048-dim embeddings
        embeddings = self.backbone(x)
        
        # 3. Reshape back to sequence
        # [batch_size, seq_len, 2048]
        embeddings = embeddings.view(batch_size, seq_len, -1)
        
        # 4. Temporal Classification
        logits = self.temporal_model(embeddings)
        
        return logits

def load_panns_weights(model, weights_path):
    """ Helper to load pretrained PANNs weights into the hybrid model's backbone """
    checkpoint = torch.load(weights_path, map_location='cpu')
    # Filter and load weights specifically for the backbone
    model_dict = model.backbone.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.backbone.load_state_dict(model_dict)
    print(f"Successfully loaded PANNs weights from {weights_path}")

if __name__ == "__main__":
    # Test
    model = HybridAudioModel()
    # 2 batch, 10 sequence, 1 channel, 100 time pts, 64 mel bins
    dummy_input = torch.randn(2, 10, 1, 100, 64)
    out = model(dummy_input)
    print(f"Hybrid Model raw output shape: {out.shape}")
