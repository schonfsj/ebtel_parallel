# To install this package
# with anaconda: "conda develop ." for development
# with pip: "pip install ." or "pip install -e ." for development
from setuptools import setup, find_packages

# List of requirements
requirements = [jupyterlab, scipy, joblib, numpy, matplotlib, python=3.8,
                astropy, jupytext, scons, seaborn]

# Package (minimal) configuration
setup(
    name = 'ebtel_parallel',
    version = 1.0.0,
    description = 'ebtel++ wrapper, creates solution grids for GX Simulator',
    packages = find_packages(),
    install_requires = requirements
)
