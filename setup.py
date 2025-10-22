# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 00:52:16 2019

@author: Jin Dou
"""

import setuptools
import re

with open("./README.md", "r", encoding='UTF-8') as fh:
  long_description = fh.read()

with open("tour/__init__.py") as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setuptools.setup(
  name="pytour",
  version=version,
  author="Powerfulbean",
  author_email="powerfulbean@gmail.com",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/powerfulbean/pytour",
  packages=setuptools.find_packages(),
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
  ],
  install_requires=[
      "mne",
      "numpy",
      "scipy",
      "matplotlib",
      "h5py"
  ],
)
