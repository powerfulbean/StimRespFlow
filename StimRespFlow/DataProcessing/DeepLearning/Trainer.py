# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:25:27 2021

@author: ShiningStone
"""
import torch
import ignite
from ignite.metrics import Loss,RunningAverage
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.contrib.handlers import ProgressBar
from ignite import handlers as igHandler
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

# fTruePredLossOutput = lambda x, y, y_pred, loss: {'true':y,'pred':y_pred,'loss':loss}
# fTruePredOutput = lambda x, y, y_pred: {'true':y,'pred':y_pred}
fPickLossFromOutput = lambda output: output['loss']
fPickPredTrueFromOutput = lambda output: (output['y_pred'],output['y'])

def fPickPredTrueFromOutputT(output):
    pred,true = output['y_pred'],output['y']
    return (pred.transpose(-1,-2),true.transpose(-1,-2))

class TCEngineOutput:
    
    def __call__(x, y, y_pred, loss = None):
        out = dict()
        out['x'] = x
        out['y'] = y
        out['y_pred'] = y_pred
        if loss:
            out['loss'] = loss
        return out 
    
def tfEngineOutput(x, y, y_pred, loss = None):
    out = dict()
    out['x'] = x
    out['y'] = y
    out['y_pred'] = y_pred
    if loss:
        out['loss'] = loss
    return out 

class CDummy:
    pass

class CTrainer:
    
    def __init__(self,epoch,device,criterion,optimizer,lrScheduler = None):
        self.curEpoch = 0
        self.nEpoch = epoch
        self.optimizer = None
        self.lrScheduler = None
        self.criterion = None
        self.device = device
        
        self.exprFlag = False
        self._historyFlag = True
        
        self.trainer = None
        self.evaluator = None
        self.model = None
        
        self.tarFolder = None
        self.oLog = None
        self.metrics = dict()
        self.metricsRecord:dict= dict()
        self.lrRecord:list = list()
        self._history:dict = {
            # 'lr':self.lrRecord,
            }
        
        self.bestEpoch = -1
        self.bestTargetMetricValue = -1
        self.targetMetric:str = None
        
        
        self.setOptm(criterion,optimizer,lrScheduler)
        self.extList:list = list()
        self.fPlotsFunc:list = list()
        
        if self._historyFlag == True:
           self.enableHistory()
    
    # def addLr(self):
    #     # print('cha yan')
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] += 0.0001
    
    # def addExpr(self):
    #     self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.addLr)
     
    def plotHistory(self):
        # print(self._history)
        fMinMax = lambda inList: \
            (inList - np.min(inList)) / (np.max(inList) - np.min(inList)) \
                if (np.max(inList) - np.min(inList)) != 0 \
                else [0.5] * len(inList)
        for key in self._history:
            self._history[key] = fMinMax(self._history[key])
            
        df = pd.DataFrame(self._history)
        sns.lineplot(data = df)
        plt.savefig(self.tarFolder + '/' + 'training_history.png')
    
    
    def enableHistory(self):
        #add plotHistory in self.targetMetric
        if self.trainer is not None:
            self.trainer.add_event_handler(Events.COMPLETED,self.plotHistory)
        pass
        
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
        if self._historyFlag:
            self._history['train_' + name] = []
            self._history['eval_' + name] = []
    
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
        for idx,param_group in enumerate(self.optimizer.param_groups):
            if self._history.get(f'lr_{idx}') is None:
                self._history[f'lr_{idx}'] = list()
            self._history[f'lr_{idx}'].append(param_group['lr'])
            self.lrRecord.append(param_group['lr'])
            
    def getLr(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
    
    def addEvaluatorExtensions(self,handler):
        # self.evaluator.add_event_handler(Events.COMPLETED, handler)
        self.extList.append([Events.COMPLETED, handler])
            
    
    def hookTrainingResults(self,trainer):
        self.plots(trainer.state.epoch)
        self.evaluator.run(self.dtldTrain)
        metrics = self.evaluator.state.metrics
        self.recordLr()
        for i in self.metricsRecord:
            val = metrics[i] 
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()
            self.metricsRecord[i]['train'].append(val)
            if self._historyFlag:
                self._history['train_' + i].append(val)
        
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
            val = metrics[i] 
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()
            self.metricsRecord[i]['eval'].append(val)
            if self._historyFlag:
                self._history['eval_' + i].append(val)
            
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
            
        
        for i in self.metrics:
            self.metrics[i].reset()
        self.evaluator.state.metrics = {}    
        torch.cuda.empty_cache()
    
    def setEvalExt(self):
        for i in self.extList:
            self.evaluator.add_event_handler(*i)
            
    def step(self):
        self.lrScheduler.step()
        # self.getLr()
    
    def setWorker(self,model,targetMetric,trainingStep = None,evaluationStep = None):
        ''' used for dowmward compatibility'''
        self._setRecording()
        self.targetMetric = targetMetric
        device = self.device
        self.model = model.to(device)
        if trainingStep:
            self.trainer = Engine(trainingStep(self,outputAdapter=tfEngineOutput))
        else:
            self.trainer = create_supervised_trainer(model,self.optimizer,self.criterion,device=device,output_transform=tfEngineOutput)
        if evaluationStep:
            self.evaluator = Engine(evaluationStep(self,outputAdapter=tfEngineOutput))
            metrics = self.metrics or {}
            for name, metric in metrics.items():
                metric.attach(self.evaluator, name)
        else:
            self.evaluator = create_supervised_evaluator(model, metrics=self.metrics,device=device,output_transform=tfEngineOutput)
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
        
        if self._historyFlag:
            self.trainer.add_event_handler(Events.COMPLETED,self.plotHistory)
        # self.addExpr()
        if isinstance(self.lrScheduler, torch.optim.lr_scheduler.CyclicLR):
            print('CyclicLR')
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.step)
            
        if isinstance(self.lrScheduler, torch.optim.lr_scheduler.OneCycleLR):
            print('OneCycleLR')
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.step)     
            # self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self._hookIterationComplete)
        # handler = EarlyStopping(patience=5, score_function=self.score_function, trainer=self.trainer)
        # self.addEvaluatorExtensions(handler)
        # self.setEvalExt()
        
    def _setRecording(self):
        for i in self.metrics:
            self.metricsRecord[i] = {'train':list(),'eval':list()}
            
    def _hookIterationComplete(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
        
    def train(self,model,targetMetric,device = 'cpu',**kwargs):
        self.setWorker(model,targetMetric,**kwargs)
        self.trainer.run(self.dtldTrain, max_epochs=self.nEpoch)
        return self.bestEpoch, self.bestTargetMetricValue
    
    def test(self,model,dtldTest,device = 'cpu',evaluationStep = None):
        # self.addMetrics('loss', Loss(self.criterion,output_transform=fPickPredTrueFromOutput))
        model = model.to(device)
        tester = CDummy()
        tester.model = model
        tester.device = device
        if evaluationStep:
            self.tester = Engine(evaluationStep(tester,outputAdapter=tfEngineOutput))
            metrics = self.metrics or {}
            for name, metric in metrics.items():
                metric.attach(self.tester, name)
        else:
            self.tester = create_supervised_evaluator(model, metrics=self.metrics,device=device,output_transform=tfEngineOutput)
        self.tester.run(dtldTest)
        metrics = self.tester.state.metrics
        print('metrics',metrics)
        return metrics

class CTrainerFunc:
    
    def __init__(self,trainer:CTrainer,outputAdapter:callable = lambda x: x):
        self.trainer = trainer
        self.outputAdapter = outputAdapter
        self.model = trainer.model
        self.device = self.trainer.device

    def __call__(self,engine,batch):
        return self.outputAdapter(*self.func(engine,batch))       
    
    def func(self,engine,batch):
        pass

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

def collate_fn_dict(batch,channelFirst = True):
    nBatch = len(batch)
    # print(len(batch))
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