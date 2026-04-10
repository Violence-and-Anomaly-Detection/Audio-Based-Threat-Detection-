import torch
import torch.nn as nn

class AudioViolenceClassifier(nn.Module):
    """
    Binary Audio Classifier for Violence vs Non-Violence.
    This architecture mimics the early layers of PANNs (CNN) but is designed 
    to be lightweight for our initial binary testing.
    """
    def __init__(self, num_classes=1): # 1 class output for binary (using Sigmoid)
        super(AudioViolenceClassifier, self).__init__()
        
        # Convolutional Feature Extractor (Processing the Log-Mel Spectrogram)
        # Input shape: (Batch, 1 Channel, n_mels=64, time_frames)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Flattens any size spectrogram to 1x1
        )
        
        # Binary Classification Head
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid() # Outputs a probability between 0.0 (Non-violence) and 1.0 (Violence)
        )

    def forward(self, x):
        # x is our Log-Mel Spectrogram image
        # x shape should be rearranged to (Batch, 1, n_mels, time_frames)
        if x.dim() == 3:
            x = x.unsqueeze(1) 
            
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Predict
        out = self.fc(x)
        return out

def test_binary_model():
    print("Testing Binary Violence Classifier with Dummy Spectrogram...")
    # Dummy Log-Mel Spectrogram image (Batch=2, Channels=1, Mels=64, TimeFrames=500)
    dummy_input = torch.randn(2, 1, 64, 500)
    
    model = AudioViolenceClassifier()
    predictions = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {predictions.shape} -> (Batch Size, Probability)")
    
    for i, prob in enumerate(predictions):
        label = "Violence" if prob.item() > 0.5 else "Non-Violence"
        print(f"Sample {i} Probability [Violence=1, Normal=0]: {prob.item():.4f} -> Prediction: {label}")

if __name__ == "__main__":
    test_binary_model()
