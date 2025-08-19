# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:33:37 2025

@author: ShiningStone
"""


def check_import(package, name):
    if package is None:
        raise ImportError(f'{name} is failed to be imported')
    return package
    