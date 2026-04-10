import torch
import torchaudio
import torch.nn.functional as F
from model import AudioViolenceClassifier
from spectrogram import LogMelSpectrogramFeatureExtractor
import os

class AudioThreatPredictor:
    def __init__(self, model_weights_path: str, target_sr=32000, duration_sec=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sr
        self.num_samples = target_sr * duration_sec
        
        # 1. Initialize Network Architecture
        self.model = AudioViolenceClassifier(num_classes=1).to(self.device)
        self.feature_extractor = LogMelSpectrogramFeatureExtractor(sample_rate=target_sr).to(self.device)
        
        # 2. Load the trained brain (.pth)
        if not os.path.exists(model_weights_path):
            print(f"Warning: Could not find model weights at '{model_weights_path}'. Make sure you downloaded it from Colab!")
            print("Running prediction with untrained (random) weights for testing purposes.")
        else:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"✅ Successfully loaded trained brain: {model_weights_path}")
            
        # 3. Set model to inference mode (disables dropout layers, etc.)
        self.model.eval()

    def process_audio(self, wav_path: str) -> torch.Tensor:
        """Loads and perfectly formats the audio file exactly like the dataset did."""
        signal, sr = torchaudio.load(wav_path)
        
        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
            
        # Convert to mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        # Pad or truncate to exact length (5 seconds)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            padding = self.num_samples - signal.shape[1]
            signal = F.pad(signal, (0, padding))
            
        # Add a "Batch" dimension because the model expects (Batch, Channels, Time)
        return signal.unsqueeze(0).to(self.device)

    def predict(self, wav_path: str):
        """Processes the audio and outputs the Threat Assessment."""
        print("-" * 50)
        print(f"Listening to: {os.path.basename(wav_path)}...")
        
        # Standardize the audio shape
        signal = self.process_audio(wav_path)
        
        # No gradients needed for predicting (saves memory and speeds up math)
        with torch.no_grad():
            # Convert to Spectrogram
            if signal.dim() == 3 and signal.size(1) == 1:
                signal = signal.squeeze(1)
            spectrogram = self.feature_extractor(signal)
            
            # Predict
            probability = self.model(spectrogram).item()
            
        # Threshold interpretation (0.5 bounds)
        threat_score = probability * 100
        if probability > 0.5:
            print(f"🚨 ALERT: Threat Detected! (Confidence: {threat_score:.1f}%)")
        else:
            print(f"✅ STATUS: Safe (Threat Confidence: {threat_score:.1f}%)")
        print("-" * 50)
        return probability

if __name__ == "__main__":
    # Test block! Your teammates can easily import the class above to their server, 
    # but you can run this script directly to test random files.
    
    WEIGHTS_FILE = "violence_audio_model.pth"
    TEST_AUDIO_FILE = "sample_test_audio.wav"  # Change this to any .wav file you want to test!
    
    predictor = AudioThreatPredictor(model_weights_path=WEIGHTS_FILE)
    
    if os.path.exists(TEST_AUDIO_FILE):
        predictor.predict(TEST_AUDIO_FILE)
    else:
        print(f"\n[Test Setup] -> Paste any audio file named '{TEST_AUDIO_FILE}' into this folder to test it!")
