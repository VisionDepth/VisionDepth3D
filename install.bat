@echo off
echo 🚀 Setting up VisionDepth3D...

:: Create Conda environment with Python 3.12
conda create -n VD3DGPU python=3.12 -y

:: Activate the new environment
call conda activate VD3DGPU

:: Install dependencies from requirements.txt
pip install -r requirements.txt

echo ✅ Setup complete! Install openCV-GPU now.
pause
