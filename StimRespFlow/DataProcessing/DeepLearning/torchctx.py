import torch

class MetricsLog:

    def __init__(self, names):
        self.data = {n: [] for n in names}

    def append(self, metricDict:dict):
        for k in metricDict:
            self.data[k].append(metricDict[k])

class Context:
    def __init__(
        self,
        model,
        metricsNames,
        folder,
        configs = {}
    ):
        self.model = model
        self.metricslog = MetricsLog(metricsNames)
        self.folder = folder
        self.configs = configs
    
class SaveBest:
    def __init__(
        self, 
        ctx:Context, 
        metricName,
        op = lambda old, new: new > old, 
        tol = None, 
    ):
        self.ctx = ctx
        self.cnt = 0
        self.bestCnt = -1
        self.metricName = metricName
        self.bestMetric = None
        self.op = op
        self.tol = tol

    @property
    def targetPath(self):
        return f'{self.ctx.folder}/saved_model.pt'

    def step(self,):
        t_metric = self.ctx.metricslog[self.metricName][-1]
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
        return ifStop