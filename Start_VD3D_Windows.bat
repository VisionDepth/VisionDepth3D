@echo off
cd /d "E:\Video Editing\VisionDepth3D-main"
call VD3D_env\Scripts\activate
python "E:\Video Editing\VisionDepth3D-main\core\GUI.py"
call VD3D_env\Scripts\deactivate
#DeleteMe2keepWindowOpen pause
#Replace Both lines in quotes with your actual path to Main folder and GUI.py
#Can delete both calls if not using a virtual environment

#thank you Ryudoadema for providing the windows startup script
