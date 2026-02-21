#!/bin/bash
# Exit on any error
set -e
echo "--- 1. Installing System Dependencies ---"
sudo apt update && sudo apt install -y sox libsox-fmt-all git wget ffmpeg

# Initialize Conda for the current shell session
# This allows 'conda activate' to work inside the script
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
echo "--- 2. Creating Conda Environment (python 3.12) ---"
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
echo "--- 3. Installing PyTorch 2.9.1 with CUDA 13.0 Support ---"
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

echo "--- 4. Installing Pre-built Flash Attention ---"
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl

echo "--- 5. Cloning Repository and Installing ---"
if [ -d "voiceclone" ]; then
    echo "Directory already exists, skipping clone..."
    cd voiceclone
else
    git clone https://github.com/treetreeder/voiceclone.git
    cd voiceclone
fi
pip install -e .
# Install in editable mode

conda create -n faster-whisper python=3.11 -y
conda activate faster-whisper
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


echo "--- Setup Complete! ---"
echo "To start working, run: conda activate qwen3-tts"