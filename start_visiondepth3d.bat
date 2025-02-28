@echo off
cd /d "%~dp0"  REM Automatically navigates to the batch file's directory

:: Check if Conda is installed
where conda >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo ✅ Conda detected. Activating environment...
    call conda activate vd3d
) ELSE (
    echo ⚠ Conda not found! Running with system Python instead...
)

:: Run VisionDepth3Dv3.py
python VisionDepth3Dv3.py

:: Pause to show any errors before closing
pause
