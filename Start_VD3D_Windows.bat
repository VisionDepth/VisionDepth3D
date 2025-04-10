@echo off
cd /d "Path\to\VisionDepth3D-main" #Update path to VisionDepth3D folder
call VD3D_env\Scripts\activate
python "D:\Video Editing\VisionDepth3D-main\core\GUI.py"
call VD3D_env\Scripts\deactivate
#DeleteMe2keepWindowOpen pause

#thank you Ryudoadema for providing the windows startup script