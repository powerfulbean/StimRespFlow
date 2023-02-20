# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:25:27 2021

@author: ShiningStone
"""
# try:
#     # from ray.air import session
#     # from ray.air.checkpoint import Checkpoint
#     from ray import tune
# except:
tune = None

import torch
import ignite
from ignite.metrics import Loss,RunningAverage
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import EarlyStopping
from ignite import handlers as igHandler
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns


# fTruePredLossOutput = lambda x, y, y_pred, loss: {'true':y,'pred':y_pred,'loss':loss}
# fTruePredOutput = lambda x, y, y_pred: {'true':y,'pred':y_pred}

def fPickLossFromOutput(output):
    return output['loss']

def fPickPredTrueFromOutput(output):
    return (output['y_pred'],output['y'])

def fIdentity(x):
    return x

# fPickLossFromOutput = lambda output: output['loss']
# fPickPredTrueFromOutput = lambda output: (output['y_pred'],output['y'])

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
    
    def __init__(self,epoch,device,criterion,optimizer,lrScheduler = None,historyFlag = True):
        self.curEpoch = 0
        self.nEpoch = epoch
        self.optimizer = None
        self.lrScheduler = None
        self.criterion = None
        self.device = device
        
        self.exprFlag = False
        self._historyFlag = historyFlag
        
        self.trainer = None
        self.evaluator = None
        self.model = None
        
        self.tarFolder = None
        self.oLog = None
        self.metrics = dict()
        self._history:dict = {
            # 'lr':self.lrRecord,
            }
        
        self.bestEpoch = -1
        self.bestTargetMetricValue = None
        self.targetMetric:str = None
        self.bestMetrics = None
        
        
        self.setOptm(criterion,optimizer,lrScheduler)
        self.extList:list = list()
        self.fPlotsFunc:list = list()
        
        self._history['iterLoss'] = []
    
    @property
    def ifEnableHistory(self):
        return self._historyFlag
     
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
        val = metrics['corr']
        return val

    
    def addPlotFunc(self,func):
        self.fPlotsFunc.append(func)
    
    def plots(self,epoch,best = False):
        if best:# or epoch % 10 == 0:
            figsList = list()
            for func in self.fPlotsFunc:
                figs = func(self.model)
                figsList += figs
            for idx, f in enumerate(figsList):
                if best:
                    f.savefig(self.tarFolder + '/' + f'best_' + str(idx) + '.png')
                else:
                    f.savefig(self.tarFolder + '/' + '_epoch_' + str(epoch) + '_' + str(idx) + '.png')
                plt.close(f)  
    
    def plotHistory(self):
        # print(self._history)
        # fMinMax = lambda inList: \
        #     (inList - np.min(inList)) / (np.max(inList) - np.min(inList)) \
        #         if (np.max(inList) - np.min(inList)) != 0 \
        #         else [0.5] * len(inList)
        # for key in self._history:
        #     self._history[key] = fMinMax(self._history[key])
        for i in self.metrics:
            fig, ax = plt.subplots()
            df = pd.DataFrame({n:self._history[n] for n in ['train_' + i, 'eval_' + i]})
            sns.lineplot(data = df,ax = ax)
            fig.savefig(self.tarFolder + '/' + f'{i}_history.png')
            plt.close(fig) 
        
        fig, ax = plt.subplots()
        df = pd.DataFrame({n:self._history[n] for n in self._history if 'lr' in n})
        sns.lineplot(data = df,ax = ax)
        fig.savefig(self.tarFolder + '/' + f'lr_history.png')
        plt.close(fig) 
        
        fig, ax = plt.subplots()
        df = pd.DataFrame({n:self._history[n] for n in ['iterLoss']})
        sns.lineplot(data = df,ax = ax)
        fig.savefig(self.tarFolder + '/' + f'iterLoss_history.png')
        plt.close(fig)
        
        if isinstance(self.lrScheduler,torch.optim.lr_scheduler.LinearLR):
            fig, ax = plt.subplots()
            lr = self._history['lr_0']
            g = sns.lineplot(x = 'lr_0', y = 'iterLoss',ax = ax,data = self._history)
            g.set(xscale="log")
            fig.savefig(self.tarFolder + '/' + f'iterLoss_vs_lr_history.png')
            plt.close(fig)
        
        
    
    def recordLr(self,):
        for idx,param_group in enumerate(self.optimizer.param_groups):
            if self._history.get(f'lr_{idx}') is None:
                self._history[f'lr_{idx}'] = list()
            self._history[f'lr_{idx}'].append(param_group['lr'])
        self._history['iterLoss'].append(self.trainer.state.metrics['loss'])
            
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
        # metrics[self.targetMetric] = np.mean(metrics[self.targetMetric])
        for i in metrics:
            metrics[i] = np.mean(metrics[i])
        
        if self._historyFlag:
            for i in self.metrics:
                val = metrics[i] 
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu()
                self._history['train_' + i].append(val)

        if self.oLog:
            self.oLog('Train','Epoch:',trainer.state.epoch,'Metrics',metrics,splitChar = '\t')
        else:
            print(f"Training Results - Epoch: {trainer.state.epoch} Metrics: {metrics}")
    
    def hookValidationResults(self,trainer):
        self.evaluator.run(self.dtldDev)
        oriMetrics = self.evaluator.state.metrics.copy()
        metrics = self.evaluator.state.metrics
        targetMetric = metrics[self.targetMetric]
        for i in metrics:
            metrics[i] = np.mean(metrics[i])
        
        if isinstance(self.lrScheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
            # print(metrics['corr'])
            self.lrScheduler.step(metrics['corr'])
        elif isinstance(self.lrScheduler,torch.optim.lr_scheduler.ExponentialLR):
            self.lrScheduler.step()
        
        if self._historyFlag:
            for i in self.metrics:
                val = metrics[i] 
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu()
                self._history['eval_' + i].append(val)
        
        if self.targetMetric == 'corr':
            fBetter = lambda cur,history: cur >= history
        elif self.targetMetric == 'loss':
            fBetter = lambda cur,history: cur <= history
        else:
            raise NotImplementedError()
        
        if self.bestTargetMetricValue is None:
            self.bestTargetMetricValue = metrics[self.targetMetric]
        
        if fBetter(metrics[self.targetMetric],self.bestTargetMetricValue):
            self.plots(trainer.state.epoch,True)
            self.bestEpoch = trainer.state.epoch
            self.bestTargetMetricValue = metrics[self.targetMetric]
            # then save checkpoint
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'targetMetric': targetMetric,
            }
            self.bestMetrics = oriMetrics#{i:oriMetrics[i].detach() for i in oriMetrics}
            torch.save(checkpoint,self.tarFolder + '/savedModel_feedForward_best.pt')
        # if self.lrScheduler:
            # print(metrics['corr'])
            # self.lrScheduler.step(metrics['corr'])
        self.getLr()
        if self.oLog:
            self.oLog('Validation','Epoch:',trainer.state.epoch,'Metrics',metrics,splitChar = '\t')
        else:
            print(f"Validation Results - Epoch: {trainer.state.epoch} Metrics: {metrics}")
            
        
        for i in self.metrics:
            self.metrics[i].reset()
        self.evaluator.state.metrics = {}    
        torch.cuda.empty_cache()

        if trainer.state.epoch  - self.bestEpoch > self.patience:
            self.trainer.terminate()
            print(f"Early stop - Epoch: {trainer.state.epoch} Metrics: {metrics} Patience: {self.patience}")
            if self.oLog:
                self.oLog(f"Early stop - Epoch: {trainer.state.epoch} Metrics: {metrics} Patience: {self.patience}")
        # return metrics[self.targetMetric]
        
        if tune is not None:
            # torch.save(
            #     (self.model.state_dict(), self.optimizer.state_dict()), self.tarFolder + "/checkpoint.pt")
            # checkpointRay = Checkpoint.from_directory(self.tarFolder)
            # session.report(loss = metrics['loss'], accuracy = metrics['corr'],checkpoint=checkpointRay)
            with tune.checkpoint_dir(trainer.state.epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)
            tune.report(loss = metrics['loss'], accuracy = metrics['corr'])
    
    def setEvalExt(self):
        for i in self.extList:
            self.evaluator.add_event_handler(*i)
            
    def step(self):
        self.lrScheduler.step()
        # self.getLr()
    
    def setWorker(self,model,targetMetric,trainingStep = None,evaluationStep = None,patience = 20):
        ''' used for dowmward compatibility'''
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
        
        # pbar = ProgressBar(persist=True,ncols = 75)
        # pbar.attach(self.trainer, metric_names= ['loss'] )
        
        # scheduler = LRScheduler(self.lrScheduler)
        # self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.reduct_step)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.hookTrainingResults)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,self.hookValidationResults)
        self.patience = patience
        # handler = EarlyStopping(patience=patience, score_function=self.hookValidationResults, trainer=self.trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        # self.evaluator.add_event_handler(Events.COMPLETED, handler)
        # self.addExpr()
        if isinstance(self.lrScheduler, torch.optim.lr_scheduler.CyclicLR):
            print('CyclicLR')
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.step)
            
        if isinstance(self.lrScheduler, torch.optim.lr_scheduler.OneCycleLR):
            print('OneCycleLR')
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.step)  

        if isinstance(self.lrScheduler,torch.optim.lr_scheduler.LinearLR):
            print('LinearLR')
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.step)  
            
        if self.ifEnableHistory:
            self.trainer.add_event_handler(Events.COMPLETED,self.plotHistory)
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED,self.recordLr)
        # handler = EarlyStopping(patience=5, score_function=self.score_function, trainer=self.trainer)
        # self.addEvaluatorExtensions(handler)
        # self.setEvalExt()
            
    def _hookIterationComplete(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
        
    def train(self,model,targetMetric,device = 'cpu',**kwargs):
        self.setWorker(model,targetMetric,**kwargs)
        self.trainer.run(self.dtldTrain, max_epochs=self.nEpoch)
        return self.bestEpoch, self.bestMetrics
    
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
        # print('metrics',metrics)
        return metrics

class CTrainerFunc:
    
    def __init__(self,trainer:CTrainer,outputAdapter:callable = fIdentity):
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