# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:39:53 2019

@author: Jin Dou
"""
from ..DataIO import checkFolder, getFileName, readAuditoryStimuli
from ..Helper.Cache import CStimuliCache
from .Abstract import CStimulus

class CWordStimulus(CStimulus):
    def configAttr()->dict:
        attr = dict()
        attr['name'] = 'CWordStimulus'
        attr['word'] = ""
        return attr
    
    def loadStimulus(self,vec):
        assert len(vec) == self.shape[0]
        self.__setitem__(slice(None), vec.reshape(-1,1))
        
    def add(self,wordStim):
        out = CWordStimulus(len(self))
        word = self.word + wordStim.word
        out = (self + wordStim)
        out.word = word
        return out
        
class CWordPhonemesStimulus(CStimulus):
    def configAttr()->dict:
        attr = dict()
        attr['name'] = 'CWordPhonemesStimulus'
        attr['word'] = ""
        attr['phs'] = []
        return attr
    
    def loadStimulus(self,vec):
        assert len(vec) == self.shape[0]
        self.__setitem__(slice(None), vec.reshape(-1,1))
        
    def add(self,wordStim):
        out = CWordPhonemesStimulus(len(self))
        word = self.word + wordStim.word
        phs = self.phs + wordStim.phs
        out = (self + wordStim)
        out.word = word 
        out.phs = phs
        return out
        
    # def configAttr():
    #     attr = dict()
    #     attr['word'] = ""
    #     return attr
        
class CVisualStimulus(CStimulus):
    pass

class CAuditoryStimulus(CStimulus):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.name = ''
        self.otherNames = list() # background stimuli name
        
    def getNFeat(self):
        return 1 + len(self.otherStreams)
    
    def loadStimulus(self,mainName:str,otherNames:list,oCache : CStimuliCache = None):
        self.otherStreams.clear()
        self.otherLen_s.clear()
        if oCache == None:
            mainStream, len_s = readAuditoryStimuli(mainName) 
            self.mainStream = mainStream
            self.len_s = len_s
            for otherStreamName in otherNames:
                otherStream, otherLen_s = readAuditoryStimuli(otherStreamName)
                self.otherStreams.append(otherStream)
                self.otherLen_s.append(otherLen_s)
        else:
            mainStream, len_s = oCache.getStimuli(mainName)
            self.mainStream = mainStream
            self.len_s = len_s
            for otherStreamName in otherNames:
                otherStream, otherLen_s = oCache.getStimuli(otherStreamName)
                self.otherStreams.append(otherStream)
                self.otherLen_s.append(otherLen_s)        
    
    def configStimuliParams(self):
        self._stimuliParams=["mainStream","otherStreams","len_s","otherLen_s"]
        self.mainStream = ''
        self.otherStreams = list([1])
        self.len_s = ''
        self.otherLen_s = list()
        
    def getSegmentStimuliLists(self,segmentLen_s,segmentsNum):#Note: for otherStreams only return first ele 
        ans = list()
        segmentMainLen_num = int(len(self.mainStream) * (segmentLen_s/self.len_s))
        segmentOtherLen_num = int(len(self.otherStreams[0]) * (segmentLen_s/self.otherLen_s[0]))
        
        rangeNum1 = int(self.len_s / segmentLen_s)
        rangeNum2 = int(self.otherLen_s[0] / segmentLen_s)
        rangeNum = min([rangeNum1,rangeNum2])
        
        if(rangeNum!=segmentsNum):
            raise ValueError("reqired segments number ",segmentsNum," is not equal to calculated segments number ",rangeNum)
        
        for i in range(rangeNum):
            tempStimuli = CAuditoryStimulus()
            tempStimuli.mainStream = self.mainStream[i*segmentMainLen_num : (i+1) * segmentMainLen_num]
            tempStimuli.otherStreams.append(self.otherStreams[0][i*segmentOtherLen_num : (i+1) * segmentOtherLen_num])
            tempStimuli.len_s = segmentLen_s
            tempStimuli.otherLen_s.append(segmentLen_s)
            ans.append(tempStimuli)
    
        return ans
    
    def getSegmentByTime(self,startTime:float,endTime:float):
#        print(startTime,endTime)
        srateM = len(self.mainStream)/self.len_s
        srateO = len(self.otherStreams[0])/self.otherLen_s[0]
        startIdx = int(startTime* srateM)
        endIdx = int(endTime* srateO)
#        print('len1',startTime,srateM,len(self.mainStream))
        tempStimuli = CAuditoryStimulus()
        tempStimuli.mainStream = self.mainStream[startIdx:endIdx]
        tempStimuli.otherStreams.append(self.otherStreams[0][startIdx:endIdx])
        tempStimuli.len_s = int(endTime- startTime)
        tempStimuli.otherLen_s.append(int(endTime - startTime))
#        print('len2',len(tempStimuli.mainStream))
        return tempStimuli
