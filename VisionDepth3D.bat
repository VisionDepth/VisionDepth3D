@echo off
cd /d "%~dp0"
echo ✅ Starting VisionDepth3D in current Conda environment: %CONDA_DEFAULT_ENV%
python core\GUI.py
pause
