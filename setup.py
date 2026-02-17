from setuptools import setup, find_packages

setup(
    name="sotif_uncertainty",
    version="1.0.0",
    description=(
        "Uncertainty evaluation methodology for SOTIF analysis "
        "of ML-based LiDAR object detection"
    ),
    author="Milin Patel, Rolf Jung",
    author_email="milin.patel@hs-kempten.de",
    url="https://github.com/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.4",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
