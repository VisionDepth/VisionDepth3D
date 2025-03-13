@echo off
echo 🚀 Starting VisionDepth3D with OpenCV CUDA support...

:: Activate the Conda environment
call conda activate VD3DGPU

:: Run the VisionDepth3D program (update with the correct script path)
python VisionDepth3D.py

echo ✅ VisionDepth3D has exited.
pause
