import torch
import os
import numpy as np
import itertools
from tqdm import tqdm
import pickle
from typing import Callable
from typing import Protocol
from torch.nn import Module as TorchModule
from torch import Tensor as TorchTensor
from torch.optim import Optimizer as TorchOptim

def dummy_transform(x):
    return x

class MetricsRecord:

    def __init__(self,):
        self._data = {}

    def append(self, metricDict:dict):
        data = self._data
        for k in metricDict:
            if k not in data:
                data[k] = []
            if isinstance(metricDict[k], list):
                data[k].extend(metricDict[k])
            else:
                data[k].append(metricDict[k])

    def newEpoch(self):
        self._data = {}

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            key, agg_func = idx
        else:
            key = idx
            agg_func = dummy_transform
        return agg_func(self._data[key])
    
    def __iter__(self):
        return iter(self._data.keys())
    
    def reduce(self, fReduce):
        output = {}
        for k in self._data:
            output[k] = fReduce(self._data[k])
        return output
    
    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self._data, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as file:
            loaded_object = pickle.load(file)
            output = cls()
            output._data = loaded_object
            return output


def result_default_transform(data):
    return torch.stack(data, 0).mean(0)

class ForwardFunc(Protocol):
    def __call__(self, model: TorchModule, batch: tuple) -> None:
        raise NotImplementedError

class BackwardFunc(Protocol):
    def __call__(
        self, 
        model: TorchModule, 
        batch: tuple,
        loss: Callable[[TorchTensor, TorchTensor], TorchTensor],
        optim: TorchOptim
    ) -> MetricsRecord:
        raise NotImplementedError

# ForwardFunc = Callable[[TorchModule, tuple], None]
# BackwardFunc = Callable[
#     [
#         TorchModule, 
#         tuple, 
#         Callable[[TorchTensor, TorchTensor], TorchTensor],
#         TorchOptim
#     ], 
#     MetricsRecord
# ]
    
class Context:
    def __init__(
        self,
        model,
        optim = None,
        folder = None,
        configs = {},
        loss = None,
        n_epochs = 100
    ):
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.model = model
        self.optim = optim
        self.loss = loss
        self.metrics_record_cache = None
        self.folder = folder
        self.configs = configs
        configs['n_epochs'] = n_epochs
    
    def new_metrics_record(self,):
        self.metrics_record_cache = MetricsRecord()
        return self.metrics_record_cache

    def fit(
        self, 
        f_forward:ForwardFunc, 
        f_backward:BackwardFunc, 
        dataloader_train, 
        dataloader_val
    ):
        n_epochs = self.n_epochs
        optim = self.optim
        loss = self.loss
        for iEpoch in range(n_epochs):
            for batch in dataloader_train:
                model = self.model.train()
                optim.zero_grad()
                f_backward(model, batch, loss, optim)
                
            metricsLog = self.new_metrics_record()

            with torch.no_grad():
                for batch in dataloader_train:
                    model = self.model.eval()
                    metrics = f_forward(model, batch)
                    metrics_new = {'train_' + k: metrics[k] for k in metrics}
                    metricsLog.append(metrics_new)

            with torch.no_grad():
                for batch in dataloader_val:
                    model = self.model.eval()
                    metrics = f_forward(model, batch)
                    metrics_new = {'val_' + k: metrics[k] for k in metrics}
                    metricsLog.append(metrics_new)

            yield metricsLog

    def evaluate(self, f_forward: ForwardFunc, dataloader):
        metricsLog = MetricsRecord()
        with torch.no_grad():
            for batch in dataloader:
                model = self.model.eval()
                metrics = f_forward(model, batch)
                metricsLog.append(metrics)
        return metricsLog


def evaluate_dataloader(model, process_batch, dtldr, fMetric, metricsLog, metric_tag = ''):
    with torch.no_grad():
        for batch in dtldr:
            model.eval()
            output = process_batch(model, batch)
            metrics:dict = fMetric(batch, output)
            if len(metric_tag) > 0:
                metrics = {f'{metric_tag}_{k}':metrics[k] for k in metrics}
            metricsLog.append(metrics)
    return metricsLog

def load_savedBest_model(saveBest:'SaveBest', fGetModel):
    model:torch.nn.Module = fGetModel()
    model.load_state_dict(saveBest.saved_checkpoint['state_dict'])
    return model

class SaveBest:
    def __init__(
        self, 
        ctx:Context, 
        metricName,
        func_reduce = dummy_transform,
        op = lambda old, new: new > old, 
        tol = None, 
        ifLog = True
    ):
        self.ctx = ctx
        self.cnt = 0
        self.bestCnt = -1
        self.metricName = metricName
        self.bestMetric = None
        self.func_reduce = func_reduce
        self.op = op
        self.tol = tol
        self.saved_checkpoint= None
        self.ifLog = ifLog

    @property
    def targetPath(self):
        return f'{self.ctx.folder}/saved_model.pt'

    def step(self,):
        t_metric = self.func_reduce(self.ctx.metrics_record_cache[self.metricName])
        t_cnt = self.cnt
        ifUpdate = False
        ifStop = False
        if self.bestMetric is None:
            ifUpdate = True
        else:
            ifUpdate = self.op(self.bestMetric, t_metric)
        if ifUpdate:
            self.bestMetric = t_metric
            self.bestCnt = t_cnt
            checkpoint = {
                'state_dict': self.ctx.model.state_dict(),
                'optim_state_dict': self.ctx.optim.state_dict(),
                'metricName': self.metricName,
                'metric': self.bestMetric,
                'cnt': self.bestCnt
            }
            checkpoint.update(self.ctx.configs)
            if self.ifLog:
                print(f'saveBest --- cnt: {self.bestCnt}, {self.metricName}: {self.bestMetric}')
            torch.save(checkpoint, self.targetPath)
            self.saved_checkpoint = checkpoint
        
        if self.tol is not None:
            if self.cnt - self.bestCnt > self.tol:
                ifStop = True
                print(f'early stop --- epoch: {self.bestCnt}, metric: {self.bestMetric}')
        
        self.cnt += 1
        return ifUpdate, ifStop