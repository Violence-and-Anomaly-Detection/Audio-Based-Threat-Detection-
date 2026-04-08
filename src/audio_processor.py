import librosa
import numpy as np
import torch

class AudioProcessor:
    def __init__(self, config):
        self.cfg = config['audio']
        
    def load_audio(self, path, duration_sec=5.0):
        """Loads and pads/truncates audio for uniformity."""
        y, sr = librosa.load(path, sr=self.cfg['sample_rate'])
        
        # Consistent duration
        target_len = int(self.cfg['sample_rate'] * duration_sec)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, max(0, target_len - len(y))))
            
        return y

    def compute_spectrogram(self, waveform):
        """Calculates the Log-Mel Spectrogram (The feature for CNN/PANNs)."""
        mel = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.cfg['sample_rate'],
            n_fft=self.cfg['n_fft'],
            hop_length=self.cfg['hop_length'],
            n_mels=self.cfg['n_mels']
        )
        # Power to Decibels (Log scaling)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel
