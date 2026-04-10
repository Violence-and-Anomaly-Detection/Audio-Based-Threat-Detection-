import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path

class ViolenceAudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Audio files and their labels.
    Automatically labels files based on filenames (e.g. 'noviolence' -> 0, anything else like 'angry' -> 1).
    """
    def __init__(self, audio_dir: str, target_sample_rate: int = 32000, max_length_seconds: int = 5):
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate
        self.max_length_seconds = max_length_seconds
        
        # We enforce a strict max length so all our tensor dimensions align.
        # e.g. 5 seconds * 32,000 samples/sec = 160,000 samples.
        self.num_samples = self.target_sample_rate * self.max_length_seconds
        
        # Automatically gather all .wav files in the directory (and subdirectories)
        self.audio_files = list(self.audio_dir.rglob("*.wav"))
        
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # 1. Get audio path
        audio_path = self.audio_files[index]
        
        # Automatically assign label based on the filename
        filename = audio_path.name.lower()
        if "noviolence" in filename:
            label = 0
        else:
            label = 1

        
        # 2. Load the audio file using torchaudio
        try:
            signal, sr = torchaudio.load(str(audio_path))
        except RuntimeError:
            # Handle corrupted files by returning a zero tensor of the correct size
            print(f"Warning: Could not load {audio_path}")
            signal = torch.zeros(1, self.num_samples)
            sr = self.target_sample_rate
            
        # 3. Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            
        # 4. Mix down to mono if stereo (2 channels)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        # 5. Pad or Truncate perfectly to our specified logic window (e.g., exactly 5 seconds)
        if signal.shape[1] > self.num_samples:
            # Truncate
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            # Pad with zeroes
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples) # pad right
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            
        return signal, torch.tensor(label, dtype=torch.long)

def get_dataloader(audio_dir: str, batch_size: int = 16, shuffle: bool = True):
    """
    Utility function to initialize and return a PyTorch DataLoader.
    """
    dataset = ViolenceAudioDataset(audio_dir=audio_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0) # Set num_workers>0 when not testing locally

if __name__ == "__main__":
    print("Dataset module ready.")
    print("Automatically assigns labels: 0 for non-violence, 1 for violence (based on filename).")
