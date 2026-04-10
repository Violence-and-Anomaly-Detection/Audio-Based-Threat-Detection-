import os
import subprocess
from pathlib import Path

def extract_audio_from_video(video_path: str, output_audio_path: str, sample_rate: int = 32000):
    """
    Extracts audio from an mp4/mkv file and saves it as a WAV file using ffmpeg.
    Resamples the audio to the target sample_rate (32kHz is standard for PANNs).
    """
    try:
        # Construct ffmpeg command: 
        # -i input, -vn (no video), -acodec pcm_s16le (16-bit PCM), -ar (sample rate), -ac 1 (mono)
        command = [
            'ffmpeg', 
            '-y', # Overwrite output files without asking
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            output_audio_path
        ]
        
        # Run the command and capture output/errors
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully extracted: {output_audio_path}")
            return True
        else:
            print(f"❌ Error extracting {video_path}:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ Error: ffmpeg is not installed or not in your system PATH.")
        print("Please download ffmpeg from https://ffmpeg.org/download.html and add it to your Windows PATH.")
        return False

def batch_extract_dataset(video_dir: str, output_dir: str):
    """
    Iterates through a directory of videos and extracts audio to the output directory.
    """
    video_dir_path = Path(video_dir)
    output_dir_path = Path(output_dir)
    
    # Ensure output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all videos (mp4, avi, mkv)
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mkv']:
        video_files.extend(list(video_dir_path.rglob(ext)))
        
    if not video_files:
        print(f"⚠️ No video files found in {video_dir}")
        return
        
    print(f"Found {len(video_files)} videos. Starting extraction...")
    
    success_count = 0
    for video in video_files:
        audio_filename = f"{video.stem}.wav"
        output_filepath = output_dir_path / audio_filename
        
        if extract_audio_from_video(str(video), str(output_filepath)):
            success_count += 1
            
    print(f"\nExtraction complete! Successfully processed {success_count}/{len(video_files)} files.")

if __name__ == "__main__":
    # Example usage:
    # Set these to your dataset paths later!
    RAW_VIDEOS = "data/raw_videos"
    PROCESSED_AUDIO = "data/processed_audio"
    
    print("Testing data ingestion script setup...")
    print("This script uses ffmpeg to reliably pull audio tracks out of MP4 files.")
    # uncomment to run:
    # batch_extract_dataset(RAW_VIDEOS, PROCESSED_AUDIO)
