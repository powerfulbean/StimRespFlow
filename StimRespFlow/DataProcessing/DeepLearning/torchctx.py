import torch
import os

class MetricsLog:

    def __init__(self, names):
        self.data = {n: [] for n in names}

    def append(self, metricDict:dict):
        for k in metricDict:
            self.data[k].append(metricDict[k])

    def __getitem__(self,key):
        return self.data[key]

def result_default_transform(data):
    return torch.stack(data, 0).mean(0)

class Result:

    def __init__(self, transform = result_default_transform):
        self.data = {}
        self.transform = transform
    
    def log(self, metricDict):
        for k in metricDict:
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(metricDict[k])
    
    def __getitem__(self, key):
        return self.transform(self.data[key])
    


class Context:
    def __init__(
        self,
        model,
        optim,
        metricsNames,
        folder,
        configs = {}
    ):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model = model
        self.optim = optim
        self.metricslog = MetricsLog(metricsNames)
        self.folder = folder
        self.configs = configs
        
    def iter_dataloader(
        self, 
        dataloader,
        func_forward,
    ):
        for batch in dataloader_train:
            

            #loss = []
            self.optim.zero_grad()
            model = self.model
            for batch in dataloader_train:
                model.train()
                # print('batch shape', batch[0].shape, batch[1].shape)
                # outputs = model(batch[0][0], batch[1][0])
                outputs = func_forward(batch)
                
                # tempLoss = pearsonr_loss(outputs[0], outputs[1])
                # tempLoss = tempLoss.sum()
                # tempLoss.backward()
                # optim.step()
                func_backward(outputs)
            
            train_cache = Result(
                lambda x: torch.stack(x, dim = 0).to('cpu').mean(0).detach()
            )
            
            
            with torch.no_grad():
                for batch in dataloader:
                    model.eval()
                    outputs2 = model(batch[0][0], batch[1][0])
                    # print(outputs[0].shape)
                    tempR = Pearsonr(outputs2[0], outputs2[1])
                    tempLoss = pearsonr_loss(outputs2[0], outputs2[1])
                    # print(tempR, tempLoss)
                    train_cache.log(
                        {
                            'train_loss': tempLoss,
                            'train_r': tempR
                        }
                    )

            val_cache = Result(
                ['val_loss', 'val_r'], 
                lambda x: torch.stack(x, dim = 0).to('cpu').mean(0).detach()
            )
            with torch.no_grad():
                for batch in test_dataloader:
                    model.eval()
                    outputs2 = model(batch[0][0], batch[1][0])
                    tempR = Pearsonr(outputs2[0], outputs2[1])
                    tempLoss = pearsonr_loss(outputs2[0], outputs2[1])
                    # print(tempR, tempLoss)
                    val_cache.log(
                        {
                            'val_loss': tempLoss, 
                            'val_r': tempR
                        }
                    )
    
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