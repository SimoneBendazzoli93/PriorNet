import glob

import setuptools
from setuptools import setup

setup(
    name="Autoinpainting",
    version="1.0",
    author="Simone Bendazzoli",
    author_email="simben@kth.se",
    description="",  # noqa: E501
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},

)
