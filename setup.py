# from skbuild import setup
from setuptools import setup
# https://scikit-build.readthedocs.io/en/latest/usage.html
setup(
    name="toycore",
    version="1.0",
    packages=['toycore'],
    cmake_source_dir='.',
)