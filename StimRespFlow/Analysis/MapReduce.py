# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:39:31 2022

@author: Jin Dou
"""
import numpy as np

def MapDictReduceList(listOfDict):
    keys = listOfDict[0].keys()
    resultsBin = {k:[] for k in keys}
    #map
    for Dict in listOfDict:
        for k,v in Dict.items():
            resultsBin[k].append(v)
            
    #reduce
    avgResults = dict.fromkeys(resultsBin)
    for k,v in resultsBin.items():
        avgResults[k] = np.mean(v,axis = 0)
    return avgResults
