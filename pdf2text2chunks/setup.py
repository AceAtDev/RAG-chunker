# setup.py
"""Setup script for PDF to RAG Converter."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="ranchunker",
    version="1.0.0",
    author="Elshoubky M (AceAtDev)",
    author_email="mohamedshoubky@gmail.com",
    description="A tool for converting PDFs to text and creating RAG-ready chunks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/aceatdev",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(), # Reads from requirements.txt, which now includes rank_bm25
    extras_require={
        "azure": [
            "azure-ai-translation-text>=1.0.0",
            "azure-storage-blob>=12.0.0",
        ],
        "google": [
            "google-cloud-vision>=3.0.0",
            "google-cloud-translate>=3.8.0",
            "google-cloud-storage>=2.7.0",
        ],
        "ocr": [
            "PyMuPDF>=1.20.0",
            "opencv-python>=4.6.0",
            "Pillow>=9.0.0",
            "pdf2image>=1.16.0",
            "pytesseract>=0.3.10",
            "easyocr>=1.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "pdf-to-rag=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)