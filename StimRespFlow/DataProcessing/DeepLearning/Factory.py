# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:27:33 2020

@author: Jin Dou
"""
import torch
import numpy as np
from ...DataStruct.DataSet import CDataSet
from ...Helper.DataObjectTransform import CNArraysToTensors

class CTorchDataset(torch.utils.data.Dataset):
    
   def __init__(self,dataset:CDataSet,forward:bool = True,device = torch.device('cpu'),T = False):
        assert isinstance(dataset,CDataSet)
        dataset.ifOldFetchMode = False
        self.dataset = dataset
        self.forward:bool = forward
        self.device = device
        self.T = T
        self.oDataTrans = CNArraysToTensors()
        
   def __getitem__(self, index):
        stimDict,resp = self.dataset[index]
        resp = self.oDataTrans(resp,T = self.T)[0]
        stimDictOut = {}
        for k,v in stimDict.items():
            if isinstance(v, np.ndarray):
                stimDictOut[k] = self.oDataTrans(v,T = self.T)[0].to(device)
            else:
                stimDictOut[k] = v
        if self.forward:
            return stimDictOut,resp
        else:
            return resp,stimDictOut


def buildDataLoader(*tensors,TorchDataSetType,oSamplerType=None,**Args):
        if(Args.get('DatasetArgs') != None):
            DataSetArgs = Args['DatasetArgs']
            dataset = TorchDataSetType(*tensors,**DataSetArgs)
        else:
            dataset = TorchDataSetType(*tensors)
        
        if(Args.get('DataLoaderArgs') != None):
            DataLoaderArgs = Args['DataLoaderArgs']
            if(oSamplerType == None or Args.get('SamplerArgs') == None):
                dataLoader = torch.utils.data.DataLoader(dataset,**DataLoaderArgs)
            else:
                SamplerArgs = Args.get('SamplerArgs')
                oSampler = oSamplerType(dataset,**SamplerArgs)
                dataLoader = torch.utils.data.DataLoader(dataset,sampler=oSampler,**DataLoaderArgs)
        else:
            dataLoader = torch.utils.data.DataLoader(dataset)
        return dataLoader
    
class CPytorch:
    
    def __init__(self):
        self.Lib = self._ImportTorch()
    
    def _ImportTorch(self):
        import torch as root
        return root
    
    def _getNNAttr(self,name:str):
        import torch.nn as NN
        ans = getattr(NN,name)
        return ans
    
class CTorchNNYaml(CPytorch):
    
    def __init__(self):
        super().__init__()
        
    def _readYaml(self,filePath):
        import yaml
        ans = None
        with open(filePath,'r') as stream:
            try:
                ans = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return ans
        
    def _ParseType(self,conf:dict):
        if(conf['Type'] == 'Sequential'):
            return self.buildSequential(conf)
        
    def _subListToTuple(self,oInput):
        if type(oInput) == dict:
            for key in oInput:
                if(type(oInput[key]) == list):
                    oInput[key] = tuple(oInput[key])
        
        elif type(oInput) == list:
            for idx,attr in enumerate(oInput):
                if type(attr) == list:
                    oInput[idx] = tuple(attr)
        
        else:
            raise ValueError("_subListToTuple: input should be dict or list")
    
    def buildSequential(self,conf:dict):
        oSeq = self.Lib.nn.Sequential()
        ModelConfList = conf['Model']
        for idx,ModelConf in enumerate(ModelConfList):
            CModule = self._getNNAttr(ModelConf[0])
            attr = ModelConf[1]
            oModule = None
            name = str(idx)
                
            if(len(ModelConf) > 2 and type(ModelConf[2]) == dict):
                '''if contain aux attribute'''
                auxAttr = ModelConf[2]
                if (auxAttr.get('name')!=None):
                    ''' if aux attribute contain name attribute'''
                    name = auxAttr['name']
            if(type(attr) == list):
                if len(attr) == 0:
                    oModule = CModule()
                elif(type(attr[0]) == list and type(attr[1]) == dict):
                    self._subListToTuple(attr[0])
                    self._subListToTuple(attr[1])
                    oModule = CModule(*attr[0],**attr[1])
                elif(any(type(x) not in [int,float,str,bool,list] for x in attr)):
                    raise ValueError('attribute of Module %s (index %d) is invalid' % (ModelConf[0],idx))
                else:
                    self._subListToTuple(attr)
                    oModule = CModule(*attr)
            elif(type(attr) == dict):
                self._subListToTuple(attr)
                oModule = CModule(**attr)
            else:
                raise ValueError('attribute of Module %s (index %d) is invalid' % (ModelConf[0],idx))
            oSeq.add_module(name,oModule)
        return oSeq
    
    def __call__(self,confFile:str):
        yamlDict = self._readYaml(confFile)
        return self._ParseType(yamlDict)
    
