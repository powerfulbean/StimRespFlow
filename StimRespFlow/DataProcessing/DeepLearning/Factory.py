# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:27:33 2020

@author: Jin Dou
"""
import torch
import numpy as np
from scipy.stats import zscore
from ...DataStruct.DataSet import CDataSet
from ...Helper.DataObjectTransform import CNArraysToTensors

def unPackDict(inDict:dict,*keys):
    out = tuple()
    for key in keys:
        out += (inDict.get(key,None),)
    return out

class CStimDataset(torch.utils.data.Dataset):
    
    def __init__(self,data,device = torch.device('cpu')):#,zscore = False
        self._data = data #list of word dict
        self.oDataTrans = CNArraysToTensors()
        self.debug = None
        self.device = device
    
    def __getitem__(self, index):
        tempDict = self._data[index]
        r1,r2,r3,r4,r7,r5,r6 = unPackDict(tempDict, 'vector','tIntvl','words','info','stimId','timeShiftCE','timeShiftSCE')
        # if self.zscore:
            # r1 = zscore(r1,axis=1)               
        
        r1,r2 = self.oDataTrans(r1,r2,T = False)
        r1 = r1.to(self.device)
        r2 = r2.to(self.device)
        if r5 is not None:
            # if self.zscore:
                # r5 = zscore(r5,axis=1)
            r5 = np.array(r5)
            r5 = self.oDataTrans(r5,T = False)[0]
            r5 = r5.to(self.device)
        output = [r1,r2,r3,r4,r7]
        output += [r5] if r5 is not None else []
        output += [r6] if r6 is not None else []
        return output
    
    def __len__(self):
        return len(self._data)

class CTorchDataset(torch.utils.data.Dataset):
    
    def __init__(self,dataset:CDataSet,forward:bool = True,device = torch.device('cpu'),T = False):#,zscore = False
        assert isinstance(dataset,CDataSet)
        dataset.ifOldFetchMode = False
        self.dataset = dataset
        self.forward:bool = forward
        self.device = device
        self.T = T
        self.oDataTrans = CNArraysToTensors()
        # self.zscore = zscore
        
    def __getitem__(self, index):
        stimDict,resp,info = self.dataset[index]
        resp = self.oDataTrans(resp,T = self.T)[0]
        stimDictOut = {}
        for k,v in stimDict.items():
            if isinstance(v, np.ndarray):
                # if self.zscore and k != 'tIntvl':
                    # v = zscore(v,axis=1)
                    #print(k,v.shape,v.mean(),v.std())
                stimDictOut[k] = self.oDataTrans(v,T = self.T)[0]#.to(self.device)
            else:
                stimDictOut[k] = v
        if self.forward:
            return stimDictOut,resp,info
        else:
            return resp,stimDictOut,info
        
    def __len__(self):
        return len(self.dataset)

    def exportStimDataset(self):
        dsNameSet = set([i.descInfo['datasetName'] for i in self.dataset.records])
        stimNumTupleList = list(set([(i.stimuli['stimKey'],i.descInfo['datasetName']) for i in self.dataset.records]))
        stimNumTupleList = sorted(stimNumTupleList)
        dsStimNumDict = {k:[] for k in dsNameSet}
        for i in stimNumTupleList:
            dsStimNumDict[i[1]].append(i)
        # stimKeys = list(set([i.stimuli['wordVecKey'] for i in self.dataset.records]))
        # print(stimKeys)
        stimList = []
        for dsKey in dsStimNumDict:
            for i in dsStimNumDict[dsKey]:
                self.dataset.stimuliDict[i[0]]['info'] = dsKey
                self.dataset.stimuliDict[i[0]]['stimId'] = i[0]
                stimList.append(self.dataset.stimuliDict[i[0]])
        return CStimDataset(stimList,device = self.device)
    

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
    
