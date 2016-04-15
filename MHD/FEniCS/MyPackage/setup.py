#!/usr/bin/env python
"""PackageName: discription of package"""

from distutils.core import setup
from setuptools import find_packages
import numpy as np

CLASSIFIERS = [
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Scientific/Engineering :: Mathematics',
'Topic :: Scientific/Engineering :: Physics',
'Operating System :: Unix',
'Operating System :: MacOS',
'Natural Language :: English',
]

setup(
    name = "PackageName",
    version = "0.0.0",
    packages = find_packages(),
    install_requires = ['numpy>=1.7',
                        'scipy>=0.13'],
    author = "Michael Wathen",
    author_email = "mwathen@cs.ubc.ca",
    classifiers=CLASSIFIERS,
    platforms = ["Linux", "Mac OS-X", "Unix"],
    use_2to3 = False,
)

