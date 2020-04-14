# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:27:33 2020

@author: Jin Dou
"""

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
    
    def buildSequential(self,conf:dict):
        oSeq = self.Lib.nn.Sequential()
        ModelConfList = conf['Model']
        for idx,ModelConf in enumerate(ModelConfList):
            CModule = self._getNNAttr(ModelConf[0])
            attr = ModelConf[1]
            oModule = None
            name = str(idx)
            if(len(ModelConf) > 2 ):
                '''if contain aux attribute'''
                auxAttr = ModelConf[2]
                if (auxAttr.get('name')!=None):
                    ''' if aux attribute contain name attribute'''
                    name = auxAttr['name']
            if(type(attr) == list):
                oModule = CModule(*attr)
            elif(type(attr) == dict):
                oModule = CModule(**attr)
            oSeq.add_module(name,oModule)
        return oSeq
    
    def __call__(self,confFile:str):
        yamlDict = self._readYaml(confFile)
        return self._ParseType(yamlDict)
        
        
        