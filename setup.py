#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

def get_version():
    # Bug Fix: absolute path so it works from any CWD
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "mangascourx", "_version.py")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.0"

def get_long_description():
    for path in ["README.md", "readme.md"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return "Advanced Multi-Scale PatchMatch & AI-Powered Text Removal Engine for Manga"

setup(
    name="mangascourx",
    version=get_version(),
    author="Zizo",
    author_email="zly30257@gmail.com",
    description="Advanced Multi-Scale PatchMatch & AI-Powered Text Removal Engine for Manga",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/zxui86/mangascourx",
    packages=find_packages(include=["mangascourx", "mangascourx.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python-headless>=4.5.0",
        "numba>=0.53.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ],
    extras_require={"dev": ["pytest>=7.0.0", "twine>=4.0.0", "build>=0.8.0"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords=["manga", "comic", "inpainting", "text-removal", "patchmatch"],
    license="MIT",
)
