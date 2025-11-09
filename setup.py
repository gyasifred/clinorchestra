#!/usr/bin/env python3
"""
Setup script for ClinOrchestra
Universal Clinical Data Extraction & Orchestration Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="clinorchestra",
    version="1.0.1",
    author="Frederick Gyasi",
    author_email="gyasi@musc.edu",
    description="Universal platform for intelligent clinical data extraction and orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clinorchestra",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "gradio==5.35.0",
        "pandas==2.3.0",
        "numpy==1.26.4",
        "torch==2.8.0",
        "transformers==4.55.4",
        "sentence-transformers==4.1.0",
        "faiss-cpu==1.12.0",
        "openai==1.93.0",
        "anthropic==0.69.0",
        "google-generativeai>=0.3.0",  
        "python-dotenv==1.1.1",
        "tqdm==4.67.1",
        "pydantic==2.11.10",
    ],
    extras_require={
        "dev": [
            "pytest==8.4.1",
            "black>=23.0.0",  
            "flake8>=6.0.0",  
            "mypy>=1.0.0",    
        ],
        "local": [
            "unsloth==2025.9.7",
            "unsloth_zoo==2025.9.9",
            "xformers==0.0.32.post2",
        ],
    },
    entry_points={
        "console_scripts": [
            "clinorchestra=annotate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)