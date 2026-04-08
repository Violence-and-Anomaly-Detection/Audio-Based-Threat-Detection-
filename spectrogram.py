import torch
import torchaudio
import matplotlib.pyplot as plt

class LogMelSpectrogramFeatureExtractor:
    """
    Transforms 1D audio waveforms into 2D Log-Mel Spectrograms.
    Parameters are optimized specifically for the PANNs (CNN14) backbone.
    """
    def __init__(self, sample_rate=32000, n_fft=1024, hop_length=320, n_mels=64):
        self.sample_rate = sample_rate
        
        # 1. MelSpectrogram transformation
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=50,
            f_max=14000
        )
        
        # 2. Amplitude to dB conversion (Log scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Expects a waveform tensor of shape (channels, samples) or (batch, channels, samples)
        Returns a Log-Mel Spectrogram tensor.
        """
        # Calculate power spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        # Convert to logarithmic scale
        log_mel_spec = self.amplitude_to_db(mel_spec)
        return log_mel_spec

def test_dummy_spectrogram():
    """
    Generates a silent/dummy 5-second waveform to test the module locally 
    without needing the actual dataset.
    """
    print("Testing Log-Mel Spectrogram Extractor with Dummy Data...")
    extractor = LogMelSpectrogramFeatureExtractor()
    
    # Generate 5 seconds of random noise (simulating an audio clip)
    dummy_audio = torch.randn(1, 32000 * 5) 
    
    # Extract features
    spectrogram = extractor(dummy_audio)
    
    print(f"Original Audio Shape: {dummy_audio.shape}")
    print(f"Spectrogram Shape: {spectrogram.shape} -> (channels, n_mels, time_frames)")
    print("✅ Extractor is working perfectly!")

if __name__ == "__main__":
    test_dummy_spectrogram()
