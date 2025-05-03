@echo off
cd /d "%~dp0"
:: Change the call under this line to the created VisionDepth3D Environment
call conda activate VD3D
echo ✅ Starting VisionDepth3D in current Conda environment: %CONDA_DEFAULT_ENV%
python VisionDepth3D.py
pause
