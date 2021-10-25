# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:25:13 2021

@author: ShiningStone
"""
import torch
from matplotlib import pyplot as plt
import numpy as np

def plotTraingRecord(records:dict):
    metrics = records['metrics']
    keys = list(metrics.keys())
    for key in keys:
        metricsVsEpoch(records, key)
    
    
def lrVSMetrics(records:dict,key):
    metrics = records['metrics']
    lr = records['lr']
    plt.figure()
    plt.plot(lr,metrics[key]['train'])
    plt.plot(lr,metrics[key]['eval'])
    
    plt.legend([f'{key}-train',f'{key}-eval'])
    plt.title(f'{key} vs lr')
    plt.xlabel('lr')
    plt.ylabel(key)
    
def metricsVsEpoch(records:dict,key):
    metrics = records['metrics']
    lr = records['lr']
    epoch = list(range(len(lr)))
    plt.figure()
    plt.plot(epoch,metrics[key]['train'])
    plt.plot(epoch,metrics[key]['eval'])
    plt.plot(epoch,lr)
    plt.legend([f'{key}-train',f'{key}-eval','lr'])
    plt.title(f'{key} lr')
    plt.xlabel('epoch')
    plt.ylabel('value')
    
def twoDatasetsEqual(set1,set2):
    assert np.all([torch.equal(set2[idx][0],i[0]) for idx,i in enumerate(set1)])
    assert np.all([torch.equal(set2[idx][1],i[1]) for idx,i in enumerate(set1)])   
    
def twoDatasetsEqualOnSpecific(set1,set2,spec):
    assert np.all([torch.equal(set2[idx][spec],i[spec]) for idx,i in enumerate(set1)])   