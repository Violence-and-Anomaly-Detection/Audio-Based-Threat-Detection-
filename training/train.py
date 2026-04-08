import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from models.hybrid_model import HybridAudioModel, load_panns_weights
from data_preprocessing.audio_dataset import AudioViolenceDataset
from tqdm import tqdm

def train():
    # 1. Load Config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Initialize Model
    model = HybridAudioModel(
        num_classes=config['model']['num_classes'],
        freeze_backbone=True # Recommended for initial training
    ).to(device)

    # Load Pretrained Weights
    weights_path = 'models/pretrained/Cnn14.pth'
    if os.path.exists(weights_path):
        load_panns_weights(model, weights_path)

    # 3. Prepare Data
    train_dataset = AudioViolenceDataset(
        processed_dir=config['data']['processed_dir'],
        seq_len=config['data']['segment_length']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=4
    )

    # 4. Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 5. Training Loop
    epochs = config['training']['epochs']
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for i, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'results/best_audio_model.pth')
            print("New best model saved!")

if __name__ == "__main__":
    train()
