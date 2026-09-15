"""
SAM3 LoRA - Standalone setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sam3-lora",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Standalone LoRA fine-tuning for SAM3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sam3_lora",
    packages=find_packages(),
    py_modules=["lora_layers"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.7.0",
        "torchvision>=0.19.0",
        "transformers>=4.48.0",
        "huggingface-hub>=0.26.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pycocotools>=2.0.6",
        "pandas>=1.5.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "submitit>=1.4.0",
        "iopath>=0.1.10",
        "decord>=0.6.0",
        "einops>=0.6.0",
        "open-clip-torch>=2.20.0",
        "openai>=1.0.0",
        "ftfy>=6.1.0",
        "regex>=2023.0.0",
        "psutil>=5.9.0",
        "torchmetrics>=1.0.0",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
