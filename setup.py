#!/usr/bin/env python3
from setuptools import setup, find_packages


tests_require = ['asynctest', 'nose']

setup(
    name="katsdpdatawriter",
    description="MeerKAT data writer",
    author="MeerKAT SDP team",
    author_email="sdpdev+katsdpdatawriter@ska.ac.za",
    packages=find_packages(),
    scripts=[
        "scripts/flag_writer.py",
        "scripts/vis_writer.py"
    ],
    setup_requires=["katversion"],
    install_requires=[
        "aiokatcp>=0.7.0",     # Needed for auto_strategy
        "spead2>=3.0.0",
        "katsdptelstate",
        "katsdpservices[argparse,aiomonitor]",
        "katdal[s3credentials]",
        "attrs",
        "aiomonitor",
        "numpy",
        "bokeh"
    ],
    extras_require={"test": tests_require},
    tests_require=tests_require,
    use_katversion=True)
