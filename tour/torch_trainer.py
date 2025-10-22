import os
import sys
import torch
import logging
import numpy as np
from itertools import chain
from typing import Callable, List, Union, Protocol

def func_reduce_mean(values):
    # print(torch.cat(values).shape)
    if values[0].ndim == 0:
        return torch.mean(torch.stack(values), dim = 0)
    else:
        return torch.mean(torch.cat(values), dim = 0)

def get_logger(
    file_dir, 
    console_level=logging.INFO, 
    file_level=logging.DEBUG, 
    file_name="logfile.log",
    if_print = True,
):
    #adopt from chat-gpt

    file_path = f"{file_dir}/{file_name}"
    logger = logging.getLogger('tray/trainer')
    logger.setLevel(logging.DEBUG)  # master level: allow all through to handlers
    logger.handlers.clear()  # prevent duplicate handlers on re-run

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if if_print:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class DependentModule(Protocol):

    def load_state(self, state:dict) -> dict:...
    def get_state(self) -> dict:...

class BatchAccumulator:

    def __init__(self,):
        self._data:list = []

    def append(self, output):
        self._data.append(output)
    
    @property
    def data(self):
        # concatenate along the batch dimension
        return torch.cat(self._data)

class MetricsRecord:

    def __init__(self,):
        self._data = {}

    def append(self, metricDict:dict, tag:str = ''):
        data = self._data
        for k in metricDict:
            if tag != '':
                real_k = tag + '/' + k
            else:
                real_k = k
            if real_k not in data:
                data[real_k] = []
            data[real_k].append(metricDict[k].cpu())
    
    def __iter__(self):
        return iter(self._data.keys())
    
    def __getitem__(self, key):
        return self._data[key]
    
    def items(self):
        for k, v in self._data.items():
            yield k,v

def ndarrays_to_tensors(*datas:List[np.ndarray]):
    # the resulted tensor will share the same memory as the array
    return [
        [
            torch.from_numpy(d) if not np.isscalar(d) else torch.tensor(d, dtype=torch.get_default_dtype()) 
            for d in data
        ] 
        for data in datas
    ]

class StimRespDataset(torch.utils.data.Dataset):

    def __init__(self, 
        stims:Union[List[np.ndarray], List[torch.Tensor]], 
        resps:Union[List[np.ndarray], List[torch.Tensor]], 
        device = 'cpu'
    ):
        if isinstance(stims[0], np.ndarray):
            stims, resps = ndarrays_to_tensors(stims, resps)
        self.stims = stims
        self.resps = resps
        self.device = device
        assert len(stims) == len(resps)

    def __getitem__(self, index:int):
        return self.stims[index].to(self.device), self.resps[index].to(self.device)

    def __len__(self):
        return len(self.stims)

class Context:

    def __init__(
        self,
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer, 
        func_metrics: Callable,
        checkpoint_folder: str, 
        checkpoint_file = "checkpoint.pt",
        custom_config = {},
        if_print_metric = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state_current_epoch = -1
        self.func_metrics = func_metrics
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_file = checkpoint_file
        self.metrics_log = MetricsRecord()
        self.custom_config = custom_config
        self.logger = get_logger(checkpoint_folder, if_print=if_print_metric)

        self.dependents:List[DependentModule] = []

    def add_dependent(self, module:DependentModule):
        self.dependents.append(module)

    def new_epochs(self):
        self.state_current_epoch += 1

    def checkpoint_exists(self):
        return os.path.exists(self.checkpoint_path)

    def save_checkpoint(self):
        checkpoint = {}
        for module in self.dependents:
            checkpoint[module.__class__.__name__] = module.get_state()
        checkpoint['context'] = self.get_state()
        torch.save(checkpoint, self.checkpoint_path)
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.load_state(checkpoint['context'])
        for module in self.dependents:
            module.load_state(checkpoint[module.__class__.__name__])
    
    @property
    def checkpoint_path(self):
        return f'{self.checkpoint_folder}/{self.checkpoint_file}'

    def log_metrics(
        self,
        metrics,
        tag = ''
    ):
        scalar_metrics = {k:v.item() for k,v in metrics.items() if v.numel() == 1}
        metrics_log = ''
        for k,v in scalar_metrics.items():
            metrics_log += f'{k}:{v} '
        self.logger.info(f"epochs:{self.state_current_epoch} - {tag} - {metrics_log}")
        self.metrics_log.append(metrics,tag)

    def new_metrics_record(self):
        return MetricsRecord()

    def evaluate_dataloader(
        self, 
        tag:str,
        dtldr:torch.utils.data.DataLoader,
        forward_function: Callable,
        f_reduce_metrics_records = func_reduce_mean,
        save_in_context = False,
        batch_hook:List[Callable] = [],
        output_hook:List[Callable] = []
    ):
        new_log = MetricsRecord()
        is_model_training = self.model.training
        with torch.no_grad():
            for batch in dtldr:
                for f_batch in batch_hook:
                    f_batch(batch)
                self.model.eval()
                output = forward_function(self.model, batch)
                for f_output in output_hook:
                    f_output(output)
                metrics_dict = self.func_metrics(
                    batch,
                    output
                )
                new_log.append(
                    metrics_dict,
                )
        # print([i.shape for i in new_log['loss']])
        # print(torch.cat(new_log['loss']).shape)
        if is_model_training:
            self.model.train()
        else:
            self.model.eval()
        
        reduced_record = {k: f_reduce_metrics_records(v) for k, v in new_log.items()}
        if save_in_context:
            self.log_metrics(reduced_record, tag)
        
        output_record = {}
        for k,v in reduced_record.items():
            if tag != '':
                real_k = tag + '/' + k
            else:
                real_k = k
            output_record[real_k] = v.cpu()
        scalar_metrics = {k:v.item() for k,v in output_record.items() if v.numel() == 1}
        return output_record, scalar_metrics

    def get_state(self):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'state_current_epoch': self.state_current_epoch,
            'custom_config':self.custom_config
        }
        return state
    
    def load_state(self, state):
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optim_state_dict'])
        self.state_current_epoch = state['state_current_epoch']
        self.custom_config = state['custom_config']

class SaveBest:
    def __init__(
        self, 
        ctx:Context,
        state_metric_name,
        op = lambda old, new: new > old, 
        tol = None, 
        ifLog = True,
        file_name = "save_best.pt"
    ):
        self.ctx = ctx
        ctx.add_dependent(self)
        self.state_cnt = 0
        self.state_best_cnt = -1
        self.state_best_metric = None
        self.state_metric_name = state_metric_name
        
        self.op = op
        self.tol = tol
        self.saved_checkpoint= None
        self.ifLog = ifLog
        self.file_name = file_name

    @property
    def target_path(self):
        return f'{self.ctx.checkpoint_folder}/{self.file_name}'

    def get_state(self):
        output = {}
        for k,v in self.__dict__.items():
            if k.startswith('state_'):
                output[k] = v
        return output
    
    def load_state(self, state):
        for k,v in self.__dict__.items():
            if k.startswith('state_'):
                self.__dict__[k] = state[k]

    def step(self,):
        t_metric = self.ctx.metrics_log[self.state_metric_name][-1]
        assert t_metric.ndim == 0 or (t_metric.ndim == 1 and t_metric.shape[0] == 1), t_metric.shape
        t_metric = t_metric.item()
        t_cnt = self.state_cnt
        ifUpdate = False
        ifStop = False
        if self.state_best_metric is None:
            ifUpdate = True
        else:
            ifUpdate = self.op(self.state_best_metric, t_metric)
        if ifUpdate:
            self.state_best_metric = t_metric
            self.state_best_cnt = t_cnt
            checkpoint = {}
            checkpoint.update(self.ctx.get_state())
            checkpoint.update(self.get_state())

            if self.ifLog:
                msg = f'save_best --- cnt: {self.state_best_cnt}, {self.state_metric_name}: {self.state_best_metric}'
                self.ctx.logger.info(msg)
            torch.save(checkpoint, self.target_path)
            self.saved_checkpoint = checkpoint
        
        if self.tol is not None:
            if self.state_cnt - self.state_best_cnt > self.tol:
                ifStop = True
                msg = f'early_stop --- epoch: {self.state_best_cnt}, metric: {self.state_best_metric}'
                self.ctx.logger.info(msg)
        self.state_cnt += 1
        return ifUpdate, ifStop
    

def pearsonr(y, y_pred):
    """
    Compute Pearson's correlation coefficient between predicted
    and observed data

    y: (..., n_samples, n_chans)
    y_pred: (..., n_samples, n_chans)
    """
    r = torch.mean(
        (y - y.mean(-2, keepdims = True)) * (y_pred - y_pred.mean(-2, keepdims = True)),
        -2
    ) / (
        y.std(-2) * y_pred.std(-2)
    )
    return r

