# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 00:52:16 2019

@author: Jin Dou
"""

import setuptools
import re

with open("./README.md", "r") as fh:
  long_description = fh.read()

with open("StimRespFlow/__init__.py") as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setuptools.setup(
  name="StimRespFlow",
  version=version,
  author="Powerfulbean",
  author_email="powerfulbean@gmail.com",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/powerfulbean/StimRespFlow",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)
