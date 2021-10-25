# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:24:16 2021

@author: ShiningStone
"""
import torch
import numpy as np
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.engine import Events

def TensorsToNumpy(*tensors):
    out = tuple()
    for tensor in tensors:
        out += (tensor.cpu().detach().numpy(),)
    return out

def Pearsonr(x,y):
    nObs = len(x)
    sumX = np.sum(x,0)
    sumY = np.sum(y,0)
    sdXY = np.sqrt((np.sum(x**2,0) - (sumX**2/nObs)) * (np.sum(y ** 2, 0) - (sumY ** 2)/nObs))    
    r = (np.sum(x*y,0) - (sumX * sumY)/nObs) / sdXY
    return r

def BatchPearsonr(pred,y,batchAvg = True):
    result = list()
    for i in range(len(pred)):
        out1 = Pearsonr(pred[i],y[i])
        result.append(out1)
        # print(np.mean(result,0).shape)
    if batchAvg:
        return np.mean(result,0)
    else:
        return result

class CMPearsonr(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu",avgOutput:bool = True, batchAvg = True):
        self._personr = None
        self._nExamples = None
        self._avgOutput:bool = avgOutput
        self._batchAvg = batchAvg
        self._personrList = list()
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._personr = 0
        self._nExamples = 0
        self._personrList = list()
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        tensors = TensorsToNumpy(y_pred,y)
        if self._batchAvg:
            self._personr += BatchPearsonr(*tensors)
            self._nExamples += 1
        else:
            self._personrList.append(BatchPearsonr(*tensors,batchAvg = self._batchAvg))
        

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        
        if not self._batchAvg:
            return np.concatenate(self._personrList,axis = 0)
        
        if self._avgOutput:
            # print('examples',self._nExamples)
            return np.mean(self._personr / self._nExamples)
        else:
            a = self._personr / self._nExamples
            print(a.shape)
            return self._personr / self._nExamples
        
    def attachForTrain(self, engine, name: str, _usage = None):
        if self.epoch_bound:
            # restart average every epoch
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

