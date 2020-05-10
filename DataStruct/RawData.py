# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 21:15:52 2019

@author: Jin Dou
"""

from .. import outsideLibInterfaces as outLib
#import six
from abc import ABC, abstractmethod
import numpy as np


class CData(ABC):
    ''' 
    rawdata: for a Data Object, it stores the true data. for a Label object, it can store the label data (audio, image, etc.).
    timeStamps: An abstract concept which is used to store the information about time and all the other necessary information
    '''
    def __init__(self):
        self.rawdata = list()
        self.timestamps = list()
        self.description = dict()
        
    def __getitem__(self,timestamp):
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
            return self._fetchBySeqIndex(slice(startIdx,endIdx,step))
                            
        else:
            idx = self.timestamps.index(timestamp)
            return self._fetchBySeqIndex(idx)
            
    def _fetchBySeqIndex(self,idx): #not based on channel
        if(type(self.rawdata[0]) != list and not isinstance(self.rawdata[0],np.ndarray)):
            return self.rawdata[idx]
        else:
            return self.rawdata[:,idx]

class CRawData(CData):
    '''
    # by default, the first dimension of self.rawdata is index of channel,
    # the second dimension of self.rawdata is timestamp
    '''
    def __init__(self): #the length of timeStamps should be as same as the number of samples 
        super(CRawData, self).__init__()
        self.startTime = '' # store the datatime.datetime object
        self.sampleRate = 0
        self.numChannels = 0
        
    def calTimeStamp(self):
        if len(self.timestamps) == 0:
            print("error: self.timestamps should be initiated in loadFile function first!")
            return
        dateTime = outLib._OutsideLibTime()._importDatetime()
        answer = list()
        answer.append( self.startTime )
        delta = dateTime.timedelta(seconds = 1/self.sampleRate)
        for idx in range(len(self.timestamps)):
            if idx == 0:
                continue
            else:
                temp = answer[idx-1] + delta
                answer.append(temp)
        
        self.timestamps = [i for i in answer]
    
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

class CBitalinoRawdata(CRawData): # EEG unit: uV; EOG unit: mv
        
    def readFile(self,filename,mode = 'EEG'):
        print("start reading bitalinofile")
        from pylab import loadtxt
        #file_name = 'opensignals_001403173836_2019-03-04_12-02-59.txt'
        fullCont = list()
        dataDescription = ''
        import json
        
        #read data description part
        with open(filename,'r') as f:
            for rowCont in f.readlines():
                if(rowCont[0] == '#' and rowCont[2] != '{'):
                   pass
                elif(rowCont[2] == '{'):
                    rowCont = rowCont[2:]
                    dataDescription = json.loads(rowCont)
                    break
                else:
                   rowArray = rowCont.split("\t")
                   rowArray = rowArray[0:-1]
                   fullCont.append(rowArray)
        
        data = loadtxt(filename)
    #    rowArrayNum = np.array(fullCont)
        rowArrayNum = data
        dateTime = outLib._OutsideLibTime()._importDatetime()
        
        for key in dataDescription.keys(): #now the key is just the mac address of the device
            dataDescription = dataDescription[key]
        
        self.timestamps = rowArrayNum[:,0]
        self.rawdata = dict()
        self.description = dataDescription
#        print(dateTime.datetime.now())
        if mode=='EEG':
            self.rawdata = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-1],10,3.3,40000,'uV')), 0)
#            self.rawdata = np.expand_dims(rowArrayNum[:,-1],0)
            self.numChannels = 1
            self.description["channelInfo"] = [[1],['EarEEG']]
        elif mode == 'EOG':
            self.rawdata = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-2],10,3.3,2040, 'mV')), 0)
            self.numChannels= 1
            self.description["channelInfo"] = [[1],['Eog']]
        elif mode == 'EEGandEOG':
            data1 = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-1],10,3.3,40000,'uV')), 0)
            data2 = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-2],10,3.3,2040, 'uV')), 0)
            self.rawdata = np.concatenate([data1,data2],0)
            self.numChannels = 2
            self.description['channelInfo'] = [[1,2],['EarEEG','Eog']]
        else:
            print("bitalino error: doesn't support this mode!")
#        print(dateTime.datetime.now())
        
        
        self.startTime = dateTime.datetime.strptime( dataDescription['date'] + ' ' + dataDescription['time'], '%Y-%m-%d %H:%M:%S.%f')
        self.sampleRate = dataDescription["sampling rate"]
        
        print("reading bitalinofile Finished")
        return data, dataDescription
    
    def getRealSignal(self,sampleDataArray, bitNumber ,VCC = 3.3 , Geeg = 40000, unit = 'uV'):
        output = [self._eegTransferFuntion(i,bitNumber ,VCC , Geeg) for i in sampleDataArray] 
        output = np.array(output)
        if(unit == 'uV'):
            output = output * (10**6)
        elif(unit == 'mV'):
            output = output * (10**3)
        return output
    
    def _eegTransferFuntion(self,sampleValue, bitNumber ,VCC, Geeg):
        output = (( (sampleValue/2**bitNumber) - 1/2) * VCC ) / Geeg
        return output
    
    
if __name__ == '__main__':
    test1 = CBitalinoRawdata()
    test2 = CRawData()
    print(test1.description)