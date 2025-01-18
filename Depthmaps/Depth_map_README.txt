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

2. If you want to generate a depth map within the program:
   - Check the "Generate Depth Maps" option in the GUI.
   - The depth map will be generated automatically and saved here.

Processing Times:
- Depth map generation times vary depending on video resolution and quality settings.
- **Estimated Time**:
  - A 30-second clip takes approximately **15-30 minutes** at standard resolutions.
  - Full-length videos (e.g., 1-2 hours) may take **5-24 hours** depending on:
    - Video resolution (e.g., 720p vs. 4K).
    - Hardware performance (CPU/GPU specifications).

Important Notes:
- Depth maps are grayscale videos that represent the distance of objects in a scene.
- Each frame of the depth map must match the resolution and frame count of the input video.

If you encounter any issues, ensure the depth map video is correctly named and located in this folder.
