from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="mlops-tapsi-ride-demand",
    version="0.2.0",
    author="Amir Jlr",
    packages=find_packages(),
    install_requires=requirements,
)
