import os
import librosa
import numpy as np
import subprocess
import yaml
from pathlib import Path
from tqdm import tqdm

class AudioProcessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sr = self.config['data']['sample_rate']
        self.n_mels = self.config['data']['n_mels']
        self.hop_size = self.config['data']['hop_size']
        self.window_size = self.config['data']['window_size']
        self.seg_len = self.config['data']['segment_length']
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.proc_dir = Path(self.config['data']['processed_dir'])
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio_from_video(self, video_path, output_wav_path):
        """
        Extracts mono audio from video file using ffmpeg.
        """
        command = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.sr), '-ac', '1',
            str(output_wav_path), '-y', '-loglevel', 'error'
        ]
        subprocess.run(command)

    def compute_mel_spectrogram(self, audio_path):
        """
        Loads audio and computes a Log-Mel Spectrogram.
        """
        y, _ = librosa.load(audio_path, sr=self.sr)
        
        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.window_size, hop_length=self.hop_size
        )
        
        # Convert to Log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def process_dataset(self):
        """
        Main loop to process all files in raw_dir.
        """
        print(f"Starting data processing from {self.raw_dir}...")
        
        # Find all video and audio files
        files = list(self.raw_dir.glob('**/*.mp4')) + list(self.raw_dir.glob('**/*.wav'))
        
        for file_path in tqdm(files):
            # 1. Handle Video to Audio conversion if needed
            if file_path.suffix == '.mp4':
                wav_temp = file_path.with_suffix('.wav')
                self.extract_audio_from_video(file_path, wav_temp)
                target_audio = wav_temp
            else:
                target_audio = file_path
            
            # 2. Extract Features
            try:
                features = self.compute_mel_spectrogram(target_audio)
                
                # 3. Save as .npy for fast loading during training
                save_path = self.proc_dir / f"{file_path.stem}.npy"
                np.save(save_path, features)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
            
            # 4. Clean up temporary wav if it was a video
            if file_path.suffix == '.mp4' and wav_temp.exists():
                os.remove(wav_temp)

if __name__ == "__main__":
    processor = AudioProcessor()
    processor.process_dataset()
