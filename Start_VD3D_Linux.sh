#!/bin/bash
cd ~/Work/VisionDepth3D #Adjust to VisionDepth3D Folder
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate VD3D
python core/GUI.py
read -p "Press Enter to exit..."
conda deactivate

#Huge Thank you to Rudi For Providing the .sh script