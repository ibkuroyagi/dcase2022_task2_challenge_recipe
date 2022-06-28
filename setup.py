#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup Anomaly Sound Detection library."""

import os

from setuptools import find_packages
from setuptools import setup

requirements = {
    "install": [],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
    "test": ["pytest>=3.3.0", "hacking>=3.0.0", "flake8-docstrings>=1.3.1"],
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="asd_tools",
    version="0.0.0",
    url="https://github.com/ibkuroyagi/dcase2022_task2.git",
    author="Ibuki Kuroyanagi",
    author_email="kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp",
    description="Anomaly Sound Detection implementation",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT License",
    packages=find_packages(include=["asd_tools*"]),
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
)
