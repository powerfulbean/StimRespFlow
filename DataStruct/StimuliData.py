# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:39:53 2019

@author: Jin Dou
"""
from abc import ABC, abstractmethod

class CStimuli(ABC):
    def __init__(self):
        self.type = ""
        self._stimuliParams = list()
        self.configStimuli()
      
    @abstractmethod
    def configStimuli(self):
        self._stimuliParams=[""]
    
    def getStimuliParams(self,show=False):
        if(show == True):
            for i in self._stimuliParams:
                print(i)
        return self._stimuliParams.copy()

class CAuditoryStimuli(CStimuli):
    
    def configStimuli(self):
        self._stimuliParams=["mainStream","otherStreams","len_s","otherLen_s"]
        self.mainStream = ''
        self.otherStreams = list()
        self.len_s = ''
        self.otherLen_s = list()
        
    def getSegmentStimuliLists(self,segmentLen_s):#Note: for otherStreams only return first ele 
        ans = list()
        segmentMainLen_num = int(len(self.mainStream) * (segmentLen_s/self.len_s))
        segmentOtherLen_num = int(len(self.otherStreams[0]) * (segmentLen_s/self.otherLen_s[0]))
        
        rangeNum1 = int(self.len_s / segmentLen_s)
        rangeNum2 = int(self.otherLen_s[0] / segmentLen_s)
        rangeNum = min([rangeNum1,rangeNum2])
        for i in range(rangeNum):
            tempStimuli = CAuditoryStimuli()
            tempStimuli.mainStream = self.mainStream[i*segmentMainLen_num : (i+1) * segmentMainLen_num]
            tempStimuli.otherStreams.append(self.otherStreams[0][i*segmentOtherLen_num : (i+1) * segmentOtherLen_num])
            tempStimuli.len_s = segmentLen_s
            tempStimuli.otherLen_s.append(segmentLen_s)
            ans.append(tempStimuli)
    
        return ans,rangeNum
    
    def getSegmentByTime(self,startTime:float,endTime:float):
#        print(startTime,endTime)
        srateM = len(self.mainStream)/self.len_s
        srateO = len(self.otherStreams[0])/self.otherLen_s[0]
        startIdx = int(startTime* srateM)
        endIdx = int(endTime* srateO)
#        print('len1',startTime,srateM,len(self.mainStream))
        tempStimuli = CAuditoryStimuli()
        tempStimuli.mainStream = self.mainStream[startIdx:endIdx]
        tempStimuli.otherStreams.append(self.otherStreams[0][startIdx:endIdx])
        tempStimuli.len_s = int(endTime- startTime)
        tempStimuli.otherLen_s.append(int(endTime - startTime))
#        print('len2',len(tempStimuli.mainStream))
        return tempStimuli
