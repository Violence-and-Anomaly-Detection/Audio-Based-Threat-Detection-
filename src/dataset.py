import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class ViolenceDataset(Dataset):
    def __init__(self, config, processor):
        self.cfg = config
        self.processor = processor
        
        # Load the Excel metadata you found earlier!
        self.df = pd.read_excel(self.cfg['paths']['vsd_excel'])
        self.audio_dir = self.cfg['paths']['vsd_audio_dir']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Construct path (matching your morning logic)
        audio_name = row['File_segment_name']
        if not audio_name.endswith('.wav'):
            audio_name += '.wav'
            
        audio_path = os.path.join(self.audio_dir, audio_name)
        
        # Load and process
        try:
            waveform = self.processor.load_audio(audio_path)
            # Label: 1 for Violence, 0 for Normal
            # Assuming 'Violence_duration' > 0 means it's a violent clip
            label = 1 if row['Violence_duration'] > 0 else 0
            
            return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            # If a file is missing, return a dummy and zero label
            return torch.zeros(160000), torch.tensor(0.0)
