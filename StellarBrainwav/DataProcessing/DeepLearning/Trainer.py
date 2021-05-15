# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:25:27 2021

@author: ShiningStone
"""
import torch
import ignite
from ignite.metrics import Loss,RunningAverage
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers import ProgressBar
from ignite import handlers as igHandler
from matplotlib import pyplot as plt
import numpy as np

fTruePredLossOutput = lambda x, y, y_pred, loss: {'true':y,'pred':y_pred,'loss':loss}
fTruePredOutput = lambda x, y, y_pred: {'true':y,'pred':y_pred}
fPickLossFromOutput = lambda output: output['loss']
fPickPredTrueFromOutput = lambda output: (output['pred'],output['true'])

def fPickPredTrueFromOutputT(output):
    pred,true = output['pred'],output['true']
    return (pred.transpose(-1,-2),true.transpose(-1,-2))



class CTrainer:
    
    def __init__(self,epoch,device,criterion,optimizer,lrScheduler = None):
        self.curEpoch = 0
        self.nEpoch = epoch
        self.optimizer = None
        self.lrScheduler = None
        self.criterion = None
        self.device = device
        
        self.trainer = None
        self.evaluator = None
        self.model = None
        
        self.tarFolder = None
        self.oLog = None
        self.metrics = dict()
        self.metricsRecord:dict= dict()
        
        self.bestEpoch = -1
        self.bestTargetMetricValue = -1
        self.targetMetric:str = None
        self.lrRecord:list = list()
        self.fPlotsFunc:list = list()
        
        self.setOptm(criterion,optimizer,lrScheduler)
        self.extList:list = list()
        
        self.exprFlag = False
    
    def addLr(self):
        # print('cha yan')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] += 0.0001
    
    def addExpr(self):
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.addLr)
        
        
    def setDataLoader(self,dtldTrain,dtldTest):
        self.dtldTrain = dtldTrain
        self.dtldDev = dtldTest
    
    def setOptm(self,criterion,optimizer,lrScheduler = None):
        self.criterion = criterion
        self.addMetrics('loss', Loss(self.criterion,output_transform=fPickPredTrueFromOutput))
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler
        
    def setDir(self,oLog,tarFolder):
        self.oLog = oLog
        self.tarFolder = tarFolder
        
    def addMetrics(self,name:str,metric:ignite.metrics.Metric):
        assert isinstance(metric,ignite.metrics.Metric)
        self.metrics[name] = metric
    
    def score_function(self,engine):
        self.evaluator.run(self.dtldDev)
        metrics = self.evaluator.state.metrics
        print('a')
        val = metrics['corr']
        return val

    def addPlotFunc(self,func):
        self.fPlotsFunc.append(func)
    
    def plots(self,epoch,best = False):
        figsList = list()
        for func in self.fPlotsFunc:
            figs = func(self.model)
            figsList += figs
        for idx, f in enumerate(figsList):
            if best:
                f.savefig(self.tarFolder + '/' + '_epoch_best_' + str(idx) + '.png')
            else:
                f.savefig(self.tarFolder + '/' + '_epoch_' + str(epoch) + '_' + str(idx) + '.png')
            plt.close(f)  
    
    def recordLr(self,):
        for param_group in self.optimizer.param_groups:
            self.lrRecord.append(param_group['lr'])
    
    def addEvaluatorExtensions(self,handler):
        # self.evaluator.add_event_handler(Events.COMPLETED, handler)
        self.extList.append([Events.COMPLETED, handler])
            
    
    def hookTrainingResults(self,trainer):
        self.plots(trainer.state.epoch)
        self.evaluator.run(self.dtldTrain)
        metrics = self.evaluator.state.metrics
        self.recordLr()
        for i in self.metricsRecord:
            self.metricsRecord[i]['train'].append(metrics[i])
        
        if self.oLog:
            self.oLog('Train','Epoch:',trainer.state.epoch,'Metrics',metrics,'lr',self.lrRecord[-1],splitChar = '\t')
        else:
            print(f"Training Results - Epoch: {trainer.state.epoch} Metrics: {metrics}")
    
    def hookValidationResults(self,trainer):
        self.evaluator.run(self.dtldDev)
        metrics = self.evaluator.state.metrics
        targetMetric = metrics[self.targetMetric]
        
        if isinstance(self.lrScheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
            # print(metrics['corr'])
            self.lrScheduler.step(metrics['corr'])
        
        for i in self.metricsRecord:
            self.metricsRecord[i]['eval'].append(metrics[i])
            
        if targetMetric > self.bestTargetMetricValue:
            self.plots(trainer.state.epoch,True)
            self.bestEpoch = trainer.state.epoch
            self.bestTargetMetricValue = targetMetric
            # then save checkpoint
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'targetMetric': targetMetric,
            }
            torch.save(checkpoint,self.tarFolder + '/savedModel_feedForward_best.pt')
        # if self.lrScheduler:
            # print(metrics['corr'])
            # self.lrScheduler.step(metrics['corr'])
        if self.oLog:
            self.oLog('Validation','Epoch:',trainer.state.epoch,'Metrics',metrics,splitChar = '\t')
        else:
            print(f"Validation Results - Epoch: {trainer.state.epoch} Metrics: {metrics}")
    
    def setEvalExt(self):
        for i in self.extList:
            self.evaluator.add_event_handler(*i)
            
    def step(self):
        self.lrScheduler.step()
    
    def setWorker(self,model,targetMetric):
        self._setRecording()
        self.targetMetric = targetMetric
        self.trainer = create_supervised_trainer(model,self.optimizer,self.criterion,output_transform=fTruePredLossOutput)
        self.evaluator = create_supervised_evaluator(model, metrics=self.metrics,output_transform=fTruePredOutput)
        self.model = model
        RunningAverage(output_transform=fPickLossFromOutput).attach(self.trainer, "loss")
        # CMPearsonr(output_transform=fPickPredTrueFromOutputT).attachForTrain(self.trainer, "corr")
        for i in self.metrics:
            if i != 'loss':
                self.metrics[i].attach(self.trainer,i)
        
        pbar = ProgressBar(persist=True,ncols = 75)
        pbar.attach(self.trainer, metric_names='all')
        
        # scheduler = LRScheduler(self.lrScheduler)
        # self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.reduct_step)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.hookTrainingResults)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.hookValidationResults)
        # self.addExpr()
        if isinstance(self.lrScheduler, torch.optim.lr_scheduler.CyclicLR):
            print('CyclicLR')
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.step)
        # handler = EarlyStopping(patience=5, score_function=self.score_function, trainer=self.trainer)
        # self.addEvaluatorExtensions(handler)
        # self.setEvalExt()
        
    def _setRecording(self):
        for i in self.metrics:
            self.metricsRecord[i] = {'train':list(),'eval':list()}
        
    def train(self,model,targetMetric):
        self.setWorker(model,targetMetric)
        self.trainer.run(self.dtldTrain, max_epochs=self.nEpoch)
        return self.bestEpoch, self.bestTargetMetricValue

def safeAllocate(arraylike):
    out = None
    if not isinstance(arraylike,torch.Tensor):
        out = torch.from_numpy(arraylike)
    else:
        out = arraylike
    return out

def collate_fn(batch,channelFirst = True):
    nBatch = len(batch)
    outBatch = []
    dimIdxLen = -1
    dimIdxChannel = -1
    if channelFirst:
        dimIdxLen = 1
        dimIdxChannel = 0
    else:
        dimIdxLen = 0
        dimIdxChannel = 1
    
    for idx,item in enumerate(batch[0]):
        maxLen = item.shape[dimIdxLen]
        nChannel = item.shape[dimIdxChannel]
        for i in range(nBatch):
            maxLen = max(batch[i][idx].shape[dimIdxLen],maxLen)
        zerosInput = [0] * 3
        zerosInput[0] = nBatch
        zerosInput[1 + dimIdxLen] = maxLen
        zerosInput[1 + dimIdxChannel] = nChannel
        tmp = torch.zeros(*zerosInput)
        for i in range(nBatch):
            indices = [0] * 3
            indices[0] = i
            indices[1 + dimIdxLen] = slice(batch[i][idx].shape[dimIdxLen])
            indices[1 + dimIdxChannel] = slice(None)
            # print(indices,i)
            indices = tuple(indices)
            # print(indices,tmp[indices].shape,batch[i][idx].shape)
            tmp[indices] = safeAllocate(batch[i][idx])
            
        outBatch.append(tmp)
    return outBatch