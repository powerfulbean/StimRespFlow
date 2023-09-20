# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:37:47 2023

@author: jdou3
"""

from enum import Enum
from dataclasses import dataclass
import torch

class Event(Enum):
    EPOCH_BEGIN = 0
    EPOCH_END = 1
    ITERATION_BEGIN = 2
    ITERATION_END = 3
    TRAIN_BEGIN = 4
    TRAIN_END = 5
    TEST_BEGIN = 6
    TEST_END = 7
    
class Handler(Enum):
    COMMON = 0
    METRICS = 1

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
    forward_output: dict
    epoch: int = 0#current epoch    
    iteration: int = 0#current iteration
    batch:tuple = (None,)
    stage:int = Stage.IDLE
    
    def reinit(self):
        self.epoch = 0
        self.iteration = 0
        self.stage = Stage.IDLE

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
        homedir = None
     ):
        self.loss = loss
        self.optim = optim
        self.forward_step = forward_step
        self.backward_step = backward_step
        
        self.epoch = epoch
        self.device = device
        self.homedir = homedir
        
        self.model:torch.nn.Module = None
        self.metrics = {'loss':loss}
        self.scheduler = None
        self.trainDataloader = None
        self.evalDataloader = None
        self.testDataloader = None
        
        self.events = {i:[] for i in Event._member_names_}
        self.add_metric('loss', self.loss, Event.EPOCH_END)
        self.engine_state = EngineState.IDLE
        self.train_state:TrainingState = TrainingState({},{})
        
    def add_scheduler(self, event, scheduler):
        self.events[event].append((Handler.COMMON, scheduler))
    
    def add_metric(self, name, metric, event = Event.EPOCH_END):
        self.metrics[name] = metric
        # self.events[event].append((Handler.METRICS, name))

    def _parseEvent(self, event):
        if event[0] == Handler.METRICS:
            name = event[1]
            metric = self.metrics[name](**self.train_state.forward_output)
            self.train_state.metrics.update({name:metric})
        elif event[0] == Handler.COMMON:
            event[1](self)
        else:
            raise ValueError()
    
    def _parseEventType(self,eventType):
        for event in self.events[eventType.name]:
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
    
    def inference(self, dataloader):
        self.model.eval()
        metrics = {i:0 for i in self.metrics}
        output = []
        cnt = 0
        for batch in dataloader:
            cnt += 1
            outputDict = self.forward_step(self, batch)
            output.append(outputDict)
            for i in metrics:
                metrics[i] += self.metrics[i](**outputDict).item()
        meanMetrics = {i:metrics[i]/cnt for i in metrics}
        self.train_state.metrics.update(meanMetrics)
        self.model.train()
        newOutput = {i:[] for i in output[0]}
        for o in output:
            for k in o:
                newOutput[k].append(o[k])
        self.train_state.forward_output.update(newOutput)
        return output
            
    
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
                    self.train_state.forward_output.update(outputDict)
                    self.backward_step(self, outputDict)
                    if self._exitIter():
                        break
                    else:
                        pass
                if self.evalDataloader is not None:
                    self.inference(self.trainDataloader)
                    print(f'train metrics {self.train_state.metrics}')
                    self.inference(self.evalDataloader)
                    print(f'evaluate metrics {self.train_state.metrics}')
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
        
        self._engine()
        self._engine(Stage.INFER)
        print(f'test metrics {self.train_state.metrics}')
        
    
    
