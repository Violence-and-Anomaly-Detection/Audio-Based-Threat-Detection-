import torch
import torchaudio
import librosa
import sys

def check_environment():
    print("=" * 40)
    print("Environment Setup Verification")
    print("=" * 40)
    print(f"Python Version: {sys.version.split(' ')[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchAudio Version: {torchaudio.__version__}")
    print(f"Librosa Version: {librosa.__version__}")
    
    print("\n" + "=" * 40)
    print("GPU / Hardware Availability")
    print("=" * 40)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU Available: True")
        print(f"✅ Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"❌ GPU Available: False")
        print("⚠️  Warning: Training deep audio models (like PANNs) on CPU will be extremely slow.")
        print("If you are running this locally, ensure you have NVIDIA drivers and PyTorch with CUDA installed.")
        print("If you are moving this to Google Colab later, you can select 'T4 GPU' from the Runtime menu.")

if __name__ == "__main__":
    check_environment()
