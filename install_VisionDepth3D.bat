@echo off
echo ===========================================
echo    VisionDepth3D Installer Script
echo ===========================================
echo.

:: Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH.
    echo Please install Anaconda or Miniconda first: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

:: Create a Conda environment
echo Creating Conda environment 'vd3d'...
conda create --name vd3d python=3.9 -y

:: Activate the environment
echo Activating Conda environment...
call conda activate vd3d

:: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Verify installation
echo.
echo ===========================================
echo    Installation Complete!
echo ===========================================
echo Use 'conda activate vd3d' before running the program.
pause
exit
