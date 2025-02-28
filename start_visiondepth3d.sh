#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

# Check if Conda is installed
if command -v conda &> /dev/null; then
    echo "✅ Conda detected. Activating environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vd3d
else
    echo "⚠ Conda not found! Running with system Python instead..."
fi

# Run VisionDepth3Dv3.py
python3 VisionDepth3Dv3.py

# Keep terminal open if there is an error
read -p "Press Enter to exit..."
