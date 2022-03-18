# Import required functions
from setuptools import setup, find_packages
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Call setup function
setup(
    author="Florence d'Alché-Buc (Researcher), Luc Motte (Researcher), Awais Sani (Engineer), Danaël  Schlewer-Becker(Engineer), Gaëtan Brison (Engineer)",
    description="Test package using IOKR method with the long term goal to develop a Structured-Prediction Package",
    name="IOKR",
    version="0.2.0",
    url="https://github.com/realpython/reader",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["pandas","numpy","scipy","scikit-learn"],
    python_requires=">=2.7"
)


