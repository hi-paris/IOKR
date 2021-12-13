# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Florence d'Alché-Buc (Researcher), Luc Motte (Researcher), Awais Sani (Engineer), Amine Yamoul (Engineer), Gaëtan Brison (Engineer)",
    description="Test package using IOKR method with the long term goal to develop a Structured-Prediction Package",
    name="IOKR",
    version="0.1.0",
    install_requires=["pandas","numpy","scipy","scikit-learn"],
    python_requires=">=2.7"
)


