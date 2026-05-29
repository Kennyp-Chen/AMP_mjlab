"""Installation script for the 'wbc_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
]

# Installation operation
setup(
    name="wbc_mjlab",
    packages=["src", "rsl_rl"]
    + [f"src.{pkg}" for pkg in find_packages("src")]
    + [f"rsl_rl.{pkg}" for pkg in find_packages("rsl_rl")],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
