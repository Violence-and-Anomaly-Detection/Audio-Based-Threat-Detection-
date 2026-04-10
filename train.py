import torch
import torch.nn as nn
import torch.optim as optim

def train_dummy_model():
    from model import AudioViolenceClassifier
    from spectrogram import LogMelSpectrogramFeatureExtractor

    print("=" * 40)
    print("Initiating Binary Violence Training Loop (Dummy Test)")
    print("=" * 40)
    
    # 1. Initialize our components
    model = AudioViolenceClassifier(num_classes=1) # Binary
    feature_extractor = LogMelSpectrogramFeatureExtractor()
    
    # Binary Cross Entropy Loss is the standard for Yes/No classification
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    
    # Simulating 1 batch of data
    # (Batch=4 audio files, 1 channel mono, 5 seconds of audio)
    dummy_audio_batch = torch.randn(4, 1, 32000 * 5)
    
    # Dummy labels: [Violence (1), Non-Violence (0), Violence (1), Non-Violence (0)]
    dummy_labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        
        # Step 1: Zero the gradients
        optimizer.zero_grad()
        
        # Step 2: Convert Audio to Spectrograms
        # Notice we squeeze the channel dimension so the extractor understands it
        spectrograms = feature_extractor(dummy_audio_batch.squeeze(1))
        
        # Step 3: Forward pass through the CNN model
        predictions = model(spectrograms)
        
        # Step 4: Calculate Error
        loss = criterion(predictions, dummy_labels)
        
        # Step 5: Backpropagation (Learn)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")
        for i, prob in enumerate(predictions):
            pred_label = "Violence" if prob.item() > 0.5 else "Non-Violence"
            ground_truth = "Violence" if dummy_labels[i].item() == 1.0 else "Non-Violence"
            print(f"  Sample {i} -> Predicted: {pred_label} ({prob.item():.2f}) | Actual: {ground_truth}")
            
    print("\n✅ Training sequence executed successfully!")
    print("The pipeline is fully connected and ready for real Google Colab data.")

if __name__ == "__main__":
    train_dummy_model()
