from setuptools import setup, find_packages

setup(
    name="osint-correlator",
    version="1.0.0",
    description="Cross-platform pseudonym correlation via behavioral fingerprinting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "torch>=2.0",
        "sentence-transformers>=2.2",
        "nltk>=3.8",
        "tqdm>=4.66",
        "rich>=13.6",
        "click>=8.1",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "osint=osint.cli:main",
        ]
    },
)
