from setuptools import setup, find_packages

setup(
    name="mallm",
    version="0.1.0",
    author="TODO",
    author_email="TODO",
    description="MALLM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jonas-becker/mallm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "fire",
        "tqdm",
        "transformers",
        "accelerate",
        "torch",
        "torchvision",
        "torchaudio",
        "bitsandbytes; platform_system == 'Linux'",  # Conditional dependency for Linux systems
    ]
)
