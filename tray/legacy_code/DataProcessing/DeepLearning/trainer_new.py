# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:37:47 2023

@author: jdou3
"""

from enum import Enum
from dataclasses import dataclass
import torch
import logging

class Event(Enum):
    EPOCH_BEGIN = 0
    EPOCH_END = 1
    ITERATION_BEGIN = 2
    ITERATION_END = 3
    TRAIN_BEGIN = 4
    TRAIN_END = 5
    TEST_BEGIN = 6
    TEST_END = 7
    EPOCH_TRAIN_INFER = 8
    EPOCH_EVAL_INFER = 9
    
class Handler(Enum):
    COMMON = 0
    SCHEDULER = 1
    
class Stage(Enum):
    IDLE = -1
    TRAIN = 0
    INFER = 1
    
class EngineState(Enum):
    STOP = 'stop'
    STOP_EPOCH = 'stop_epoch'
    IDLE = 'idle'
    START = 'start'

class RunningStage:
    
    _stage = Stage.IDLE
    
    def train(self):
        self._stage = Stage.TRAIN
        
    def eval(self):
        self._stage = Stage.EVAL
        
    def test(self):
        self._stage = Stage.TEST
        
    def __call__(self):
        return self._stage

@dataclass
class TrainingState:
    
    metrics: dict #
    # forward_output: dict
    epoch: int = 0#current epoch    
    iteration: int = 0#current iteration
    batch:tuple = (None,)
    stage:int = Stage.IDLE
    
    def reinit(self):
        self.epoch = 0
        self.iteration = 0
        self.stage = Stage.IDLE

class FuncHandler:

    def __call__(self, oTrainer:'TorchTrainer'):
        return self.step(oTrainer)

    def step(self,oTrainer:'TorchTrainer'):
        pass

class AddOn(FuncHandler):

    default_event = Event.EPOCH_EVAL_INFER

class SaveBest(AddOn):
    def __init__(self, metricName,ifLarger,tol = None):
        self.metricName = metricName
        self.bestEpoch = -1
        self.bestMetric = None
        self.ifLarger = ifLarger
        self.tol = tol

    def step(self, oTrainer):
        t_metric = oTrainer.train_state.metrics[self.metricName]
        t_epoch = oTrainer.train_state.epoch
        ifUpdate = False
        if self.bestMetric is None:
            ifUpdate = True
        else:
            if self.ifLarger:
                ifUpdate = t_metric > self.bestMetric
            else:
                ifUpdate = t_metric < self.bestMetric
        if ifUpdate:
            self.bestMetric = t_metric
            self.bestEpoch = t_epoch
            checkpoint = {
                'state_dict': oTrainer.model.state_dict(),
                'targetMetric': self.bestMetric,
                'epoch': self.bestEpoch
            }
            oTrainer.logger.info(f'saveBest --- epoch: {self.bestEpoch}, metric: {self.bestMetric}')
            torch.save(checkpoint, f'{oTrainer.folder}/saved_model.pt')
        
        if self.tol is not None:
            if oTrainer.train_state.epoch - self.bestEpoch > self.tol:
                oTrainer.engine_state = EngineState.STOP
                oTrainer.logger.info(f'early stop --- epoch: {self.bestEpoch}, metric: {self.bestMetric}')

class TorchTrainer:
    
    #design philosophy, organize stats into a dictionary
    #forward_step should use dict as output
    #backward_step should receive dict as input
    
    def __init__(self, 
        loss, 
        optim,
        forward_step,
        backward_step,
        epoch = 100, 
        device = torch.device('cpu'), 
        folder = None
     ):
        self.loss = loss
        self.optim = optim
        self.forward_step = forward_step
        self.backward_step = backward_step
        
        self.epoch = epoch
        self.device = device
        self.folder = folder
        
        self.model:torch.nn.Module = None
        self.metrics = {'loss':loss}
        self.metricToLog:set = set()
        self.scheduler = None
        self.trainDataloader = None
        self.evalDataloader = None
        self.testDataloader = None
        
        self.events = {i:[] for i in list(Event)}
        self.add_metric('loss', self.loss, Event.EPOCH_END)
        self.engine_state = EngineState.IDLE
        self.train_state:TrainingState = TrainingState({})
        
        logger = logging.getLogger('trainer')
        for h in logger.handlers:
            logger.removeHandler(h)
        
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        if folder is not None:
            ch = logging.FileHandler(f'{folder}/log.txt')
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        self.logger = logger
        
    def add_scheduler(self, eventType, scheduler):
        self.add_event(eventType, Handler.SCHEDULER, scheduler)
    
    def add_event(self, eventType:Event,handlerType:Handler, func):
         self.events[eventType].append((handlerType, func))

    def add_metric(self, name, metric,ifPrint = True):
        self.metrics[name] = metric
        if ifPrint:
            self.metricToLog.add(name)
        # self.events[event].append((Handler.METRICS, name))

    def _parseEvent(self, event):
        if event[0] == Handler.COMMON:
            event[1](self)
        elif event[0] == Handler.SCHEDULER:
            event[1].step()
        else:
            raise ValueError()
    
    def _parseEventType(self,eventType):
        for event in self.events[eventType]:
            self._parseEvent(event)
    
    def _enterEngine(self):
        self.train_state.reinit()
        self.engine_state = EngineState.START
        assert self.model is not None
    
    def _exitEngine(self):
        self.engine_state = EngineState.IDLE
    
    def _enterTest(self):
        self.model.eval()
        self.train_state.stage = Stage.INFER
        self._parseEventType(Event.TEST_BEGIN)
        
    def _exitTest(self):
        self.model.eval()
        self.train_state.stage = Stage.IDLE
        self._parseEventType(Event.TEST_END)
    
    def _enterTrain(self):
        self.model.train()
        self.train_state.stage = Stage.TRAIN
        self._parseEventType(Event.TRAIN_BEGIN)
        
    def _exitTrain(self):
        self.model.eval()
        self.train_state.stage = Stage.IDLE
        self._parseEventType(Event.TRAIN_END)

    def _enterEpoch(self):
        self._parseEventType(Event.EPOCH_BEGIN) #exe epoch begin
        self.train_state.epoch += 1

    def _exitEpochTrainVal(self):
        self._parseEventType(Event.EPOCH_TRAIN_INFER)

    def _exitEpochEvalVal(self):
        self._parseEventType(Event.EPOCH_EVAL_INFER)

    def _exitEpoch(self):
        ''' similar as the exitIter'''
        self._parseEventType(Event.EPOCH_END)
        self.train_state.iteration = 0
        if self.engine_state == EngineState.STOP_EPOCH:
            return 'continue'
        elif self.engine_state == EngineState.STOP:
            return True
        else:
            return False
    
    def _enterIter(self):
        self._parseEventType(Event.ITERATION_BEGIN)
        self.train_state.iteration += 1
    
    def _exitIter(self):
        ''' will return if break the loop '''
        self._parseEventType(Event.ITERATION_END)
        if self.engine_state in [EngineState.STOP and EngineState.STOP_EPOCH]:
            return True
        else:
            return False
    
    def detachOutput(self,outputdict):
        for k in outputdict:
            outputdict[k] = outputdict[k].detach()


    def customInference(self, dataloader, forward_step):
        self.model.eval()
        with torch.no_grad():
            outputDicts = []
            for batch in dataloader:
                outputDict = forward_step(self, batch)
                outputDicts.append(outputDict)
            keys = list(outputDicts[0].keys())
            output = {k:[] for k in keys}
            for o in outputDicts:
                for k in keys:
                    output[k].append(o[k])
        return output

    def inference(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            metrics = {i:0 for i in self.metrics}
            # output = []
            cnt = 0
            for batch in dataloader:
                outputDict = self.forward_step(self, batch)
                nBatch = list(outputDict.values())[0].shape[0]
                self.detachOutput(outputDict)
                # output.append(outputDict)
                for i in metrics:
                    metrics[i] += self.metrics[i](**outputDict) * nBatch
                cnt += nBatch
            meanMetrics = {i:(metrics[i]/cnt).cpu().numpy() for i in metrics}
            for k in meanMetrics:
                if meanMetrics[k].shape == (1,) or \
                    meanMetrics[k].ndim == 0:
                    meanMetrics[k] = meanMetrics[k].item()
            self.train_state.metrics.update(meanMetrics)
        self.model.train()
        # newOutput = {i:[] for i in output[0]}
        # for o in output:
        #     for k in o:
        #         newOutput[k].append(o[k])
        # self.train_state.forward_output.update(newOutput)
        # return output
        return meanMetrics
            
    def logMetric(self,tag = ''):
        output = {}
        for metric in self.metricToLog:
            output[metric] = self.train_state.metrics[metric]
        self.logger.info(f'{tag} - {output}')

    
    def _engine(self, mode = Stage.TRAIN):
        #train first
        self._enterEngine()
        if mode == Stage.TRAIN:
            assert self.trainDataloader is not None
            self._enterTrain()
            for iEpoch in range(self.epoch):
                self._enterEpoch()        
                for batch in self.trainDataloader:
                    self._enterIter()
                    outputDict = self.forward_step(self, batch)
                    # self.train_state.forward_output.update(outputDict)
                    self.backward_step(self, outputDict)
                    if self._exitIter():
                        break
                    else:
                        pass
                if self.evalDataloader is not None:
                    self.inference(self.trainDataloader)
                    self.logMetric('train metrics ')
                    self._exitEpochTrainVal()
                    self.inference(self.evalDataloader)
                    self.logMetric('evaluate metrics ')
                    self._exitEpochEvalVal()
                flag = self._exitEpoch()
                if flag == True:
                    break
                elif flag == 'continue':
                    continue
                else:
                    pass
            self._exitTrain()
        
        if mode == Stage.INFER:
            assert self.testDataloader is not None
            self._enterTest()
            self.inference(self.testDataloader)
            self._exitTest()
        self._exitEngine()
    
    def run(self, 
            model, 
            trainDataloader = None, 
            evalDataloader = None, 
            testDataloader = None,
            mode = Stage.TRAIN):
        
        model = model.to(self.device)
        self.model = model
        self.trainDataloader = trainDataloader
        self.evalDataloader = evalDataloader
        self.testDataloader = testDataloader
        
        if trainDataloader is not None:
            self._engine()
        if testDataloader is not None:
            self._engine(Stage.INFER)
        self.logger.info(f'test metrics {self.train_state.metrics}')
        
    def addOn(self, addon:AddOn, event = None):
        assert isinstance(addon, AddOn)
        if event is None:
            event = addon.default_event
        self.add_event(event, Handler.COMMON, addon)

    
