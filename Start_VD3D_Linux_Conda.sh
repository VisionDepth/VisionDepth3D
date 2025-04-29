#!/bin/bash
cd ~/Work/VisionDepth3D  # Adjust this to match your VisionDepth3D path

# Load conda into the shell (more portable)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "❌ Conda not found! Please install Miniconda or Anaconda."
    exit 1
fi

conda activate VD3D
python core/GUI.py

read -p "Press Enter to exit..."
conda deactivate
