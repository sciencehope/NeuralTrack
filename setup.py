# setup.py
from setuptools import setup, find_packages
import os

# Function to read the contents of the README file
def read_readme():
    with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
        return f.read()

# Read the README content for long description
long_description = read_readme()

# Read version from the package
def get_version():
    version = {}
    with open("neuraltrack/__init__.py") as f:
        # Go through each line and find the line defining __version__
        for line in f:
            if line.startswith("__version__"):
                # Extract the version number from the line
                exec(line, version)
                break
    return version["__version__"]

setup(
    name='neuraltrack',
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "numpy", "matplotlib"
    ],
    author="Science Hope",
    author_email="",
    description="NeuralTrack is a lightweight logging tool for deep learning training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/sciencehope/NeuralTrack",  # Change to your GitHub repo
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "neuraltrack-plot-loss = neuraltrack.visualization.loss_plot:main",
            "neuraltrack-plot-gradient = neuraltrack.visualization.gradient_plot:main"
        ]
    },
)
