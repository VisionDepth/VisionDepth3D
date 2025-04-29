@echo off
cd /d "E:\Video Editing\VisionDepth3D-main"

:: Activate the local virtual environment (VD3D_env)
call VD3D_env\Scripts\activate.bat

:: Start the GUI
python "core\GUI.py"

:: Optional: deactivate environment after closing
call VD3D_env\Scripts\deactivate.bat

pause

:: NOTES:
:: - Replace "E:\Video Editing\VisionDepth3D-main" with your actual folder path if different.
:: - Only keep the 'activate'/'deactivate' lines if you are using a virtual environment.
:: - If using base Anaconda/Conda environment, you should 'call conda activate VD3D' instead.
:: - Special thanks to Ryudoadema for helping provide the starter template!

