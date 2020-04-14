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
    
    def __init__(self,confFile):
        super().__init__()
        self.yamlDict = self._readYaml(confFile)
        
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
            self.buildTorchSequential(conf)
    
    def buildSequential(conf:dict):
        pass
    
        
        
        