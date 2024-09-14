import torch
import os
import numpy as np
import itertools
from tqdm import tqdm

def dummy_transform(x):
    return x

fSelectTrials = lambda data, iTrials: [[r[i] for i in iTrials] for r in data]
fSelectSubjs = lambda data, iSubjs: [data[i] for i in iSubjs]
fFlatten = lambda data: list(itertools.chain.from_iterable(data))

def get_datasets_subjectCV(
        data,
        prepare_dataset,
        iFold, 
        nFolds, 
        iTrialsForTrain, 
        iTrialsForVal,
        CDataset: torch.utils.data.dataset.Dataset,
        iTrialsForTest = None
    ):
    #prepare_dataset: #output shape [nSubjs, nTrials] list of numpy array
    data = prepare_dataset(data)
    split_iFold =  np.array_split(np.arange(nFolds), nFolds)
    iFold = iFold
    iSubjs_trainVal = np.concatenate(split_iFold[:iFold] + split_iFold[iFold + 1:])
    iSubjs_test = split_iFold[iFold]
    
    trainValData = fSelectSubjs(data, iSubjs_trainVal)
    trainData = fFlatten(fSelectTrials(trainValData, iTrialsForTrain))
    valData = fFlatten(fSelectTrials(trainValData, iTrialsForVal))

    testData = fSelectSubjs(data, iSubjs_test)
    if iTrialsForTest is not None:
        testData = fFlatten(fSelectTrials(testData, iTrialsForTest))
    else:
        testData = fFlatten(testData)
            
    return (
        torch.utils.data.DataLoader(CDataset(trainData)), 
        torch.utils.data.DataLoader(CDataset(valData)), 
        torch.utils.data.DataLoader(CDataset(testData))
    )

class MetricsLog:

    def __init__(self, agg_func = dummy_transform):
        self._data = [{}]
        self.agg_func = agg_func

    def append(self, metricDict:dict):
        data = self._data[-1]
        for k in metricDict:
            if k not in data:
                data[k] = []
            if isinstance(metricDict[k], list):
                data[k].extend(metricDict[k])
            else:
                data[k].append(metricDict[k])

    def newEpoch(self):
        self._data[-1] = {}

    def __getitem__(self, idx):
        key, index = idx
        return self.agg_func(self._data[index][key])

def result_default_transform(data):
    return torch.stack(data, 0).mean(0)
    
class Metrics:

    def __init__(self, func):
        self._func = func

    def __call__(self, output, tag = '') -> dict:
        metrics =  self._func(output)
        assert isinstance(metrics, dict)
        if len(tag) > 0:
            new_metrics = {f'{tag}_{k}':metrics[k] for k in metrics}
        else:
            new_metrics = metrics
        return new_metrics

class Context:
    def __init__(
        self,
        model,
        optim = None,
        folder = None,
        configs = {}
    ):
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.model = model
        self.optim = optim
        self.metricslog = MetricsLog()
        self.folder = folder
        self.configs = configs
        self._curEpoch = -1
    
    @property
    def nEpoch(self):
        return len(self.metricslog._data)

    def model_infer_dataloader(
        self, 
        dataloader,
        func_forward,
        curEpoch = 0
    ):
        with torch.no_grad():
            model = self.model
            if curEpoch != self._curEpoch:
                self.metricslog.newEpoch()
                self._curEpoch = curEpoch
            for batch in dataloader:
                model.eval()
                outputs = func_forward(model, batch)
                yield outputs, self.metricslog
                
    def model_dataloader_input_output(
        self,
        dataloader,
        func_forward,
        curEpoch = 0,
    ):
        with torch.no_grad():
            model = self.model
            if curEpoch != self._curEpoch:
                self.metricslog.newEpoch()
                self._curEpoch = curEpoch
            for batch in tqdm(dataloader):
                model.eval()
                output = func_forward(model, batch)
                yield batch, output, self.metricslog

    def decorator_model_op(self,func):
        def wrapper(*args, **kwargs):
            return func(self.model, *args, **kwargs)
        return wrapper
    
    def create_metrics_func(self, func) -> Metrics:
        return Metrics(func)


class SaveBest:
    def __init__(
        self, 
        ctx:Context, 
        metricName,
        func_reduce = dummy_transform,
        op = lambda old, new: new > old, 
        tol = None, 
    ):
        self.ctx = ctx
        self.cnt = 0
        self.bestCnt = -1
        self.metricName = metricName
        self.bestMetric = None
        self.func_reduce = func_reduce
        self.op = op
        self.tol = tol

    @property
    def targetPath(self):
        return f'{self.ctx.folder}/saved_model.pt'

    def step(self,):
        t_metric = self.func_reduce(self.ctx.metricslog[self.metricName,-1])
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
            print(f'saveBest --- cnt: {self.bestCnt}, {self.metricName}: {self.bestMetric}')
            torch.save(checkpoint, self.targetPath)
        
        if self.tol is not None:
            if self.cnt - self.bestCnt > self.tol:
                ifStop = True
                print(f'early stop --- epoch: {self.bestCnt}, metric: {self.bestMetric}')
        
        self.cnt += 1
        return ifUpdate, ifStop