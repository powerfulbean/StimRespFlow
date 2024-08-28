import torch
import os

def dummy_transform(x):
    return x

class MetricsLog:

    def __init__(self, nEpoch, agg_func = dummy_transform):
        self._data = [{} for i in range(nEpoch)]
        self._nEpoch = nEpoch
        self.agg_func = agg_func

    def append(self, iEpoch, metricDict:dict):
        data = self._data[iEpoch]
        for k in metricDict:
            if k not in data:
                data[k] = []
            data[k].append(metricDict[k])

    def __getitem__(self,key, index):
        return self.agg_func(self.data[index][key])

def result_default_transform(data):
    return torch.stack(data, 0).mean(0)
    
class FuncForward:

    def __init__(self, model, func):
        self._model = model
        self._func = func
    
    def __call__(self, batch):
        return self._func(self._model, batch)

class Metrics:

    def __init__(self, func):
        self._func = func

    def __call__(self, output, tag = '') -> dict:
        metrics =  self._func(output)
        assert isinstance(metrics, dict)
        if len(tag) > 0:
            new_metrics = {f'{tag}_k':metrics[k] for k in metrics}
        else:
            new_metrics = metrics
        return new_metrics

class Context:
    def __init__(
        self,
        nEpoch,
        model,
        optim,
        folder,
        configs = {}
    ):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model = model
        self.optim = optim
        self.metricslog = MetricsLog(nEpoch)
        self.folder = folder
        self.configs = configs
        
    def model_infer_dataloader(
        self, 
        dataloader,
        func_forward,
    ):
        with torch.no_grad():
            model = self.model
            for batch in dataloader:
                model.eval()
                outputs = func_forward(batch)
                yield outputs

    def decorator_model_op(self,func):
        def wrapper(*args, **kwargs):
            return func(self.model, *args, **kwargs)
        return wrapper
    
    def create_forward_func(self, func) -> FuncForward:
        return FuncForward(self.model, func)
    
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