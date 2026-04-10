import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_dataloader
from model import AudioViolenceClassifier
from spectrogram import LogMelSpectrogramFeatureExtractor

def train_model(audio_dir: str, epochs: int = 5, batch_size: int = 16, lr: float = 0.001):
    print("=" * 50)
    print("Initiating Binary Violence Training Loop")
    print(f"Data Source: {audio_dir}")
    print("=" * 50)
    
    # 1. Setup Device (MPS for Mac, CUDA for Nvidia/Colab, CPU as fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Initialize DataLoader using our updated dataset.py
    print("Loading dataset... (This may take a minute)")
    dataloader = get_dataloader(audio_dir=audio_dir, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded! Total batches per epoch: {len(dataloader)}")
    
    # 3. Initialize Model and Feature Extractor
    model = AudioViolenceClassifier(num_classes=1).to(device)
    feature_extractor = LogMelSpectrogramFeatureExtractor().to(device)
    
    # 4. Setup Loss and Optimizer
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (audio_signals, labels) in enumerate(dataloader):
            # Move data to GPU if available
            audio_signals = audio_signals.to(device)
            # BCELoss expects float labels, shaped like (batch, 1)
            labels = labels.to(device).float().unsqueeze(1)
            
            # Step 1: Zero the gradients
            optimizer.zero_grad()
            
            # Step 2: Convert Audio to Spectrograms on the fly
            # Squeeze channel dim for extractor if necessary
            if audio_signals.dim() == 3 and audio_signals.size(1) == 1:
                audio_signals = audio_signals.squeeze(1)
            spectrograms = feature_extractor(audio_signals)
            
            # Step 3: Forward pass
            predictions = model(spectrograms)
            
            # Step 4: Calculate Error
            loss = criterion(predictions, labels)
            
            # Step 5: Backpropagation
            loss.backward()
            optimizer.step()
            
            # Tracking stats
            running_loss += loss.item()
            
            # Metrics
            predicted_classes = (predictions > 0.5).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
            
            # Print update every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        # Epoch Summary
        epoch_accuracy = (correct_predictions / total_samples) * 100
        epoch_loss = running_loss / len(dataloader)
        print("-" * 50)
        print(f"End of Epoch {epoch+1} | Avg Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
        print("-" * 50)
            
    # 6. Save the trained weights (.pth file)
    weights_path = "violence_audio_model.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"\n✅ Training Complete! Model saved as '{weights_path}'")

if __name__ == "__main__":
    # IMPORTANT: We set this to the path where we extracted the data in Google Drive!
    DATA_DIRECTORY = "/content/drive/MyDrive/Audio-Based-Threat-Detection-/datasets/Audio-Violence-Dataset/audios_VSD/audios_VSD"
    
    # Start training!
    train_model(audio_dir=DATA_DIRECTORY)
