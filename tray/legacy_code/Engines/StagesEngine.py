# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 00:54:46 2021

@author: Jin Dou
"""

'''
Stages Protocol

Output in the previous step, will be used as input for the next step
The output  = [item1,item2] 
where item1 is argument list, and item2 is keyword argument dict  

'''

from abc import ABC, abstractmethod


class CComponentInterface(ABC):
    
    def __init__(self, name, returnDict):
        self.returnDict = returnDict
        self.name = name
   
    @abstractmethod
    def do(self,):
        return None
    
    def __call__(self,*args, **kwargs):
        out = self.do(*args, **kwargs)
        assert all([i in out for i in self.returnDict])
        return out
    
    def log(self,):
        #will use the logging interface
        pass

_middleOutputCache = []

def addMiddleResult(args,kwargs):
    _middleOutputCache.append((args,kwargs))
    
def fetchMiddleResult():
    return _middleOutputCache.pop(0)


class CStageSeqItem:
    def __init__(self,func,args = [],kwargs = {}):
        self.func = func
        self.args = args
        self.kwargs = kwargs

#funcSeqList is a dictionary of dicts of StageSeqItem
# each list item contains a function reference
funcSeqList = dict()
paramsLoadFlag = False

# def registerFuncForStage(stage:str):
#     def registerFunc(func):
#         def wrapper(*args,**kwargs):
#             global funcSeqList
#             if stage not in funcSeqList:
#                 funcSeqList[stage] = list()
#             funcSeqList[stage].append((func,args,kwargs))
#             return func(*args,**kwargs)
#         return wrapper
#     return registerFunc
def clearList():
    global funcSeqList
    global _middleOutputCache
    funcSeqList.clear()
    _middleOutputCache.clear()

def stage(stage:str,seqId:int):
    assert type(stage) == str
    assert type(seqId) == int
    def registerFunc(func):
        global funcSeqList
        if stage not in funcSeqList:
            funcSeqList[stage] = dict()
        funcSeqList[stage][seqId] = CStageSeqItem(func)
        def wrapper(*args,**kwargs):
            funcSeqList[stage][seqId].args = args
            funcSeqList[stage][seqId].kwargs = kwargs
            if paramsLoadFlag: 
                #In this mode, the function will not really run,
                #but just register its param with the StagesEngine
                return None
            else:
                return func(*args,**kwargs)
        return wrapper
    return registerFunc

def startEngine(stageSeq = None):
    if not stageSeq:
        stageSeq = funcSeqList.keys()
        
    for stage in stageSeq:
        for idx in sorted(funcSeqList[stage].keys()):
            item:CStageSeqItem = funcSeqList[stage][idx]
            func = item.func
            args = item.args
            kwargs = item.kwargs
            if len(args) > 0 or len(kwargs) > 0:
                output = func(*args,**kwargs)
            else:
                param = fetchMiddleResult()
                output = func(*param[0],**param[1])
            if output is not None:
                addMiddleResult(output[0], output[1])
            

class CStagesEngine:
    
    def __init__(self,stageSeq = None):
        self.stageSeq = stageSeq
    
    def __enter__(self):
        global paramsLoadFlag
        paramsLoadFlag = True
        return self
        
    def __exit__(self,exc_type, exc_val, exc_tb):
        global paramsLoadFlag
        paramsLoadFlag = False
        
    def startEngine(self,stageSeq = None):
        if stageSeq:
            startEngine(stageSeq)
        else:
            startEngine(self.stageSeq)
    
    def clearList(self):
        clearList()
    



