import os
import yaml
import librosa
import numpy as np
import torch
import subprocess

class AudioProcessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.env = self.config['environment']
        self.paths = self.config['paths'][self.env]
        self.audio_cfg = self.config['audio']
        
        # Ensure directories exist
        os.makedirs(self.paths['raw_audio_dir'], exist_ok=True)
        os.makedirs(self.paths['processed_dir'], exist_ok=True)

    def extract_audio_from_video(self, video_path, output_filename):
        """
        Uses ffmpeg to extract audio from a video file.
        """
        output_path = os.path.join(self.paths['raw_audio_dir'], output_filename)
        command = [
            'ffmpeg', '-i', video_path, 
            '-ab', '160k', '-ac', '1', 
            '-ar', str(self.audio_cfg['sample_rate']), 
            '-vn', output_path, '-y'
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return output_path
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return None

    def load_audio(self, audio_path):
        """
        Loads audio file and resamples if necessary.
        """
        waveform, sr = librosa.load(audio_path, sr=self.audio_cfg['sample_rate'])
        return waveform

    def compute_melspectrogram(self, waveform):
        """
        Computes Log-Mel Spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.audio_cfg['sample_rate'],
            n_fft=self.audio_cfg['window_size'],
            hop_length=self.audio_cfg['hop_size'],
            n_mels=self.audio_cfg['mel_bins'],
            fmin=self.audio_cfg['fmin'],
            fmax=self.audio_cfg['fmax']
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

if __name__ == "__main__":
    processor = AudioProcessor()
    print(f"Environment: {processor.env}")
    print(f"Raw Audio Dir: {processor.paths['raw_audio_dir']}")
