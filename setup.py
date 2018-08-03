#!/usr/bin/env python3
from setuptools import setup, find_packages


setup(
    name="katsdpdatawriter",
    description="MeerKAT data writer",
    author="SARAO",
    author_email="spt@ska.ac.za",
    packages=find_packages(),
    scripts=["scripts/flag_writer.py"],
    setup_requires=["katversion"],
    install_requires=[
        "aiokatcp>=0.3.0",     # Needed for status_func
        "spead2>=1.8.0",       # Needed for async iteration
        "katsdptelstate",
        "katsdpservices",
        "katdal",
        "attrs",
        "numpy"
    ],
    use_katversion=True)