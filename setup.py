from setuptools import setup, find_packages

setup(
    name="VisionDepth3D",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "tensorflow",
        "pillow",
        "tqdm",
        "tk",  # Only if using Tkinter
        "matplotlib",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vd3d=VisionDepth3D.main:main",  # Adjust this based on your main script
        ],
    },
)
