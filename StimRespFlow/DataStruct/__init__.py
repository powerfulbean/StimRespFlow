# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 01:12:07 2019

@author: Jin Dou
"""
import numpy as np
def featDictProcess(func):
    def wrapper(stimDict, srcKeys, tarsKeys):
        for k in stimDict:
            if isinstance(srcKeys, str):
                srcKeys = [srcKeys]
            if isinstance(tarsKeys, str):
                tarsKeys = [tarsKeys]
            result = func(*[stimDict[k][src] for src in srcKeys])
            if len(tarsKeys) == 1:
                result = [result]
            for idx,tar in enumerate(tarsKeys):
                stimDict[k][tar] = result[idx] 
        return stimDict
    return wrapper

def get_toImpulses(fs, padding_s = 0):
    def wrapper(vectors, timestamps):
        return toImpulses(vectors, timestamps, fs, padding_s)
    return wrapper

def toImpulses(vectors, timestamps, f:float,padding_s = 0):
    '''
    # align the vectors into impulses with specific sampling rate 
    '''
    startTimes = timestamps[0]
    endTimes = timestamps[1]
    secLen = endTimes[-1] + padding_s
    nLen = np.ceil( secLen * f).astype(int)
    nDim = vectors.shape[0]
    out = np.zeros((nDim,nLen))
    
    timeIndices = np.round(startTimes * f).astype(int)
    out[:,timeIndices] = vectors
    return out