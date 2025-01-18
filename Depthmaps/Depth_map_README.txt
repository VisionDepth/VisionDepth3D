Depthmaps Folder
----------------

This folder stores depth map videos used for 3D rendering.

Naming Convention:
- Depth maps must follow the naming pattern:
  depth_<input_video_name>.mp4
  Example: For input video `my_video.mp4`, the depth map should be named:
  depth_my_video.mp4

How to Use:
1. If you already have a depth map:
   - Place it in this folder.
   - Ensure it follows the correct naming convention.

Important Notes:
- Depth maps are grayscale videos that represent the distance of objects in a scene.
- Each frame of the depth map must match the resolution and frame count of the input video.

If you encounter any issues, ensure the depth map video is correctly named and located in this folder.
