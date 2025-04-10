@echo off
cd /d "%~dp0"
call conda activate VD3D #change to created VisionDepth3D Environment
echo ✅ Starting VisionDepth3D in current Conda environment: %CONDA_DEFAULT_ENV% 
python core\GUI.py
pause

#Huge thank you to Aether for adding line 3 to finish the script
