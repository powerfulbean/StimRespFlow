# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 00:52:16 2019

@author: Jin Dou
"""

import setuptools

with open("./README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="StellarBrainwav",
  version="0.0.1",
  author="Powerfulbean",
  author_email="powerfulbean@gmail.com",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/StellarBlocks/StellarInfra",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)