@echo off
echo ===========================================
echo    VisionDepth3D Installer Script
echo ===========================================
echo.

:: Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo Detected Conda. Setting up in a Conda environment...
    goto conda_setup
) else (
    echo Conda not found. Using standard Python installation...
    goto pip_setup
)

:: Setup with Conda
:conda_setup
echo Creating Conda environment 'vd3d'...
conda create --name vd3d python=3.9 -y

echo Activating Conda environment...
call conda activate vd3d

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===========================================
echo    Installation Complete (Conda)!
echo ===========================================
echo Use 'conda activate vd3d' before running the program.
pause
exit

:: Setup with Standard Python
:pip_setup
echo Installing dependencies globally or in your virtual environment...
python -m venv vd3d_env
call vd3d_env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===========================================
echo    Installation Complete (Standard Python)!
echo ===========================================
echo Use 'vd3d_env\Scripts\activate' before running the program.
pause
exit
