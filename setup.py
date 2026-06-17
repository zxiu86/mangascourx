# setup.py
from setuptools import setup, find_packages

setup(
    name="MangaScourX",
    version="1.0.0",
    author="Zizo",
    description="Advanced Multi-Scale PatchMatch & AI-Powered Text Removal Engine for Manga",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zxui86/MangaScourX",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "numba>=0.53.0",
        "torch>=1.9.0",
        "torchvision"
    ],
)
