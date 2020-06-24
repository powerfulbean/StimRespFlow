# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:55:28 2020

@author: Jin Dou
"""

class CStageControl:
    
    def __init__(self,tartgetList,oLog=None):
        self.targetList = tartgetList
        self.oLog = oLog
        
    def stage(self,stageNum):
        if(stageNum in self.targetList):
            if(self.oLog != None):
                self.oLog.safeRecord('Stage_' + str(stageNum) + ':')
            return True
        else:
#            sys.exit(0)
            return False
        
    def __call__(self,stageNum):
        return self.stage(stageNum)

stageList = [0]

def configStage(List):
    global stageList
    stageList = List

def decrtr_stage(stageNum):
    def decorator(func):
        def wrapper(*args, **kw):
            if stageNum in stageList:
                return func(*args, **kw)
            else:
                return False
        return wrapper
    return decorator
