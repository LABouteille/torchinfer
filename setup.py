from setuptools import find_packages, setup

setup(
    name="torchinfer",
    version="0.1",
    author="Ferdinand Mom",
    description=" Deep learning inference framework [WIP] ",
    packages=find_packages(exclude=["sandbox", "torchinfer", "targets"]),
    python_requires=">=3.8",
)