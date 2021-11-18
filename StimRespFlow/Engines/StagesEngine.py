# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 00:54:46 2021

@author: Jin Dou
"""

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
    funcSeqList.clear()

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
            func(*args,**kwargs)

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
    



