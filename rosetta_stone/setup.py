"""
Setup script for AI Rosetta Stone Engine
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-rosetta-stone",
    version="0.1.0",
    author="AI Rosetta Stone Team",
    author_email="contact@ai-rosetta-stone.org",
    description="Neuro-Symbolic Engine for AI Regulatory Compliance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-rosetta-stone/ai-rosetta-stone",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Other/Nonlisted Topic",  # Legal compliance
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rosetta-stone=rosetta_stone.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rosetta_stone": [
            "data/*.json",
            "data/*.owl",
            "reporting/templates/*.html",
            "reporting/templates/*.css",
        ],
    },
    zip_safe=False,
)