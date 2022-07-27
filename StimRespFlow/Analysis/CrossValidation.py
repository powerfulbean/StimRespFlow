# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:26:30 2022

@author: Jin Dou
"""


from sklearn.model_selection import KFold

def genKFold(stim,resp,nSplits):
    for trainIdx,testIdx in KFold(n_splits=nSplits).split(stim):
        stimTrain = [stim[i] for i in trainIdx]
        stimTest  = [stim[i] for i in testIdx]
        respTrain = [resp[i] for i in trainIdx]
        respTest  = [resp[i] for i in testIdx]
        yield stimTrain,stimTest,respTrain,respTest

