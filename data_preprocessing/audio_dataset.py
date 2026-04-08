import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class AudioViolenceDataset(Dataset):
    def __init__(self, processed_dir, labels_csv=None, seq_len=10, transform=None):
        """
        Custom Dataset for Audio Violence Detection.
        - processed_dir: Path to folder with .npy files
        - labels_csv: CSV file mapping filename to label index
        - seq_len: Number of 1-second chunks per sample
        """
        self.processed_dir = Path(processed_dir)
        self.seq_len = seq_len
        self.transform = transform
        
        # Load labels if provided, else list all .npy files (inference mode)
        if labels_csv and os.path.exists(labels_csv):
            self.labels_df = pd.read_csv(labels_csv)
            self.file_list = self.labels_df['filename'].tolist()
            self.labels = self.labels_df['label'].tolist()
        else:
            self.file_list = [f.name for f in self.processed_dir.glob('*.npy')]
            self.labels = [0] * len(self.file_list) # Dummy labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.processed_dir / self.file_list[idx]
        
        # 1. Load the spectrogram (Log-Mel)
        # Assuming the saved .npy is [F, T]
        data = np.load(file_path)
        
        # 2. Sequential segmenting logic 
        # For simplicity, if data is long, we take the first seq_len chunks
        # In a real scenario, you might want to sliding window here
        # Here we assume each .npy is already prepared or we crop/pad
        
        # Convert to tensor and add channel dim [C, T, F]
        data_tensor = torch.from_numpy(data).float()
        
        # Basic padding/cropping to ensure consistent time dimension if needed
        # Expected shape per chunk: [1, Time, Mels]
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0) # [1, F, T]
            data_tensor = data_tensor.transpose(1, 2) # [1, T, F]
            
        # If we are doing sequence modeling, we need [Seq, C, T, F]
        # For this skeleton, we assume each file is 1 sequence.
        # Check if we need to expand into 'seq_len' chunks
        if data_tensor.shape[0] != self.seq_len:
            # Simple repeat for dummy demonstration; 
            # Real research logic would split a long clip into seq_len parts
            data_tensor = data_tensor.repeat(self.seq_len, 1, 1, 1)

        label = torch.tensor(self.labels[idx]).long()
        
        return data_tensor, label

if __name__ == "__main__":
    # Test
    # dataset = AudioViolenceDataset('data/processed')
    print("Dataset module initialized.")
