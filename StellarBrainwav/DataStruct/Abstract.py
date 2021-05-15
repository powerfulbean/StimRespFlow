# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:03:21 2021

@author: Jin Dou
"""

from abc import ABC, abstractmethod
import numpy as np
from .Array import CStimuliVectors,CStimulusVector
from ..DataIO import checkFolder
from ..Helper.Cache import CStimuliCache

class CTimeStampsGen(ABC):
    def __init__(self,start,delta,nLen):
        self.start = start
        self.delta = delta
        self.nLen = nLen
        self.state = start
        self.idx = 0
    
    def __iter__(self):
        return self
    
    @abstractmethod
    def __next__(self):
        pass
        

class CData(ABC):
    ''' 
    rawdata: for a Data Object, it stores the true data. for a Label object, it can store the label data (audio, image, etc.).
    timeStamps: An abstract concept which is used to store the information about time and all the other necessary information
    
    '''
    def __init__(self):
        self._data = None
        self.timestamps:list = list()
        self.description = dict()
    
    @property    
    def rawdata(self):
        assert self.dataCheck(self._data)
        return self._data
    
    @rawdata.setter
    def rawdata(self,data):
        assert self.dataCheck(data)
        self._data = data
        
    @property    
    def data(self):
        assert self.dataCheck(self._data)
        return self._data
    
    @data.setter
    def data(self,data):
        assert self.dataCheck(data)
        self._data = data
    
    @abstractmethod
    def dataCheck(self,data):
        return True
    
    def row(self,row):
        return self.rawdata[row,:]
    
    def col(self,col):
        return self.rawdata[:,col]
    
    def __getitem__(self,timestamp):
        idx = self._seqIDToSeqIdx(timestamp)
        return self._fetchBySeqIndex(idx)
    
    def _seqIDToSeqIdx(self,timestamp):
        if(isinstance(timestamp,slice)):
            startIdx = 0
            endIdx = None
            step = timestamp.step
            if(type(step) != int and step != None):
                raise ValueError("CData: slice's step should be an integer")
            if(timestamp.start == None and timestamp.stop != None):
                stopTimeId = timestamp.stop
                endIdx = self.timestamps.index(stopTimeId)
            elif(timestamp.stop == None and timestamp.start != None):
                startTimeId = timestamp.start
                startIdx = self.timestamps.index(startTimeId)
            elif(timestamp.start != None and timestamp.stop != None):
                startTimeId = timestamp.start
                stopTimeId = timestamp.stop
                startIdx = self.timestamps.index(startTimeId)
                endIdx = self.timestamps.index(stopTimeId)
#            print(type(step))
            return slice(startIdx,endIdx,step)
                            
        else:
            idx = self.timestamps.index(timestamp)
            return idx
            
    def _fetchBySeqIndex(self,idx): #not based on channel
        if isinstance(self.rawdata,np.ndarray):
            return self.rawdata[:,idx]
        else:
            return self.rawdata[idx]
        


class CRawData(CData):
    '''
    # by default, the first dimension of self.rawdata is index of channel,
    # the second dimension of self.rawdata is timestamp
    '''
    def __init__(self): #the length of timeStamps should be the same as the number of samples 
        super().__init__()
        self.startTime = '' # store the datatime.datetime object
        self.sampleRate = 0
        self.numChannels = 0
    
    def dataCheck(self,data:np.ndarray):
        out = True
        assert isinstance(data,np.ndarray)
        assert len(data.shape) == 2
        assert len(self.timestamps) == data.shape[1]
        return out
            
    def calTimeStamp(self,timeStampsGenerator:CTimeStampsGen):
        self.timestamps = [i for i in timeStampsGenerator]
    
    def sorted_key(self,x):
        return x.startTime
    
    def findInterval(self,startTime,endTime,frontLag_s, postLag_s):
        '''
        return the copy of the rawdata with the interval satisfying the requirement most
        '''
        if( endTime < self.startTime or startTime > self.timestamps[-1]):
            return None,False
        else:
            headerIdx = 0
            tailIdx = 0
            for i in range(len(self.timestamps)): #find the header index with the time just equal or ahead the startTime
                if(startTime <= self.timestamps[i]):
                    if(startTime == self.timestamps[i] or i==0):
                        headerIdx = i
                    else:
                        headerIdx = i - 1
                    break
            
            tailIdx = headerIdx
            
            for i in range(headerIdx,len(self.timestamps)): #find the tail index with the time just equal or behind the endTime
                if(endTime >= self.timestamps[i]):
                    if(endTime == self.timestamps[i] or i==len(self.timestamps)-1):
                        tailIdx = i
                    else:
                        tailIdx = i + 1
                else:
                    break
            
            headerIdx -= int(frontLag_s * self.sampleRate)
            tailIdx += int(postLag_s * self.sampleRate)
            
            ## check if the index is out of bound
            if(headerIdx < 0):
                headerIdx = 0
            
            if(tailIdx > len(self.timestamps)):
                tailIdx = len(self.timestamps) - 1

            return self.rawdata[:,headerIdx:tailIdx],(self.timestamps[headerIdx],self.timestamps[tailIdx])
    
    @abstractmethod
    def readFile(self,fileName):
        ''' abstract method for file reading'''
        pass

class CStimulusMixin(ABC):
    
    @abstractmethod
    def configAttr()->dict:
        attr = dict()
        attr['name'] = 'CStimulusMixin'
        return attr
        
    @abstractmethod
    def loadStimulus(self):
        pass
    

#derive from numpy array
class CStimulus(CStimulusMixin,CStimulusVector):
    """
    __len__ return the number of features
    
    we force the _data's first dimension to indicate features, 
    and second dimension to indicate length.
    """
    
    def __new__(cls,shape,*args,**kwargs):
        if hasattr(cls, '__abstractmethods__') and len(cls.__abstractmethods__) > 0:
            raise TypeError(f"Can't instantiate abstract class {cls.__name__} with abstract methods {', '.join(cls.__abstractmethods__)}")
        return super().__new__(cls,shape)
    # @abstractmethod
    # def getSegmentStimuliLists(self,segmentLen_s,segmentsNum):
    #     pass
    
    def getStimuliParams(self,show=False):
        if(show == True):
            for i in self._stimuliParams:
                print(i)
        return self._stimuliParams.copy()
    
    
    # @abstractmethod
    # def getNFeat(self):
    #     pass

class CLabels(CData): 
    # for labels created by psychopy, the last number is microsecond but not millisecond 
    '''
    By default, we will create a 'CLabelDataList' object for self._data, however
    please feel free to use other list/array like data struct to store the features
    of your label(i.e. stimuli)
    
    we force the _data's first dimension to indicate features, 
    and second dimension to indicate length.
    '''
    def __init__(self,nFeat):
        super().__init__()
        self.startTime = None
        self._data = CStimuliVectors(nFeat) #list()
        self._nFeat = nFeat
    
    @property
    def nFeat(self):
        return self._nFeat
    
    @nFeat.setter
    def nFeat(self,v):
        self._nFeat = v
        self._data.nFeat = v
    
    def writeInfoForMatlab(self,folder):
        import json
        ans = dict()
        ans["startDate"] = str(self.startTime)
        dataDict = dict()
        cnt = 0
        for i in self.timestamps:
            s,e = i.getLabelDict() # get start and end event time
            dataDict[str(cnt)] = s
            cnt+=1
            dataDict[str(cnt)] = e
            cnt+=1 
        ans["labels"] = dataDict
        app_json = json.dumps(ans)
        #print(app_json)
        checkFolder(folder)
        fileAddress = str(folder) + self.description+"_labels_" + str(self.startTime).replace(":","-") + ".jin" 
        f = open(fileAddress,'w+')
        f.write(app_json)
        f.close
    
    def writeType(self,typeName):
        for i in self.timestamps:
            i.type = typeName
    
    def dataCheck(self,data):
        assert self.nFeat == len(data)
        if hasattr(data, 'shape'):
            assert len(self.timestamps) == data.shape[1]
        else:
            assert len(self.timestamps) == len(data[0])
        return True
    
    def append(self,timeStamp,stimuli):
        self.rawdata.append(stimuli)
        self.timestamps.append(timeStamp)
    
    @abstractmethod
    def readFile(self,fileName):
        return
    
    @abstractmethod
    def loadStimuli(self,Folder, extension, oCache : CStimuliCache = None):
        ''' 
        load stimuli in self.timestamps(CLabelInfoCoarse).stimuli
        '''
        pass