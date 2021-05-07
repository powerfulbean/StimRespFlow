# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:25:27 2021

@author: ShiningStone
"""
import torch
import ignite
from ignite.metrics import Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers import ProgressBar
from matplotlib import pyplot as plt

fTruePredOutput = lambda x, y, y_pred, loss: {'true':y,'pred':y_pred,'loss':loss}
fPickLossFromOutput = lambda output: output['loss']
fPickPredTrueFromOutput = lambda output: (output['pred'],output['true'])

def fPickPredTrueFromOutputT(output):
    pred,true = output['pred'],output['true']
    return (pred.transpose(-1,-2),true.transpose(-1,-2))

class CTrainer:
    
    def __init__(self,epoch,device,dtldTrain,dtldTest,fExtVar):
        self.fExtVar = fExtVar
        self.curEpoch = 0
        self.nEpoch = epoch
        self.optimizer = None
        self.lrScheduler = None
        self.criterion = None
        self.device = device
        self.dtldTrain = dtldTrain
        self.dtldDev = dtldTest
        self.tarFolder = None
        self.oLog = None
        self.metrics = dict()
        
        self.trainer = None
        self.evaluator = None
        
    
    def setOptm(self,criterion,optimizer,lrScheduler = None):
        self.criterion = criterion
        self.addMetrics('loss', Loss(self.criterion,output_transform=fPickLossFromOutput))
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler
        
    def setDir(self,oLog,tarFolder):
        self.oLog = oLog
        self.tarFolder = tarFolder
        
    def addMetrics(self,name:str,metric:ignite.metrics.Metric):
        assert isinstance(metric,ignite.metrics.Metric)
        self.metrics[name] = metric
    
    def hookTrainingResults(self,trainer):
        self.evaluator.run(self.dtldTrain)
        metrics = self.evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch} Metrics: {metrics}")
    
    def hookValidationResults(self,trainer):
        self.evaluator.run(self.dtldDev)
        metrics = self.evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch} Metrics: {metrics}")
        
    def setWorker(self,model):
        from ignite.contrib.handlers.param_scheduler import LRScheduler
        self.trainer = create_supervised_trainer(model,self.optimizer,self.criterion)
        self.evaluator = create_supervised_evaluator(model, metrics=self.metrics)
        
        pbar = ProgressBar()
        pbar.attach(self.trainer, metric_names='all')
        
        scheduler = LRScheduler(self.lrScheduler)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.hookTrainingResults)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.hookValidationResults)
        
        
    def train(self,model):
        self.setWorker(model)
        self.trainer.run(self.dtldTrain, max_epochs=self.nEpoch)