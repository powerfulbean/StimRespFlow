# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 23:21:06 2021

@author: ShiningStone
"""

import datetime
import numpy as np
from .Abstract import CWaveData,CTimeStampsGen

class CDateTimeStampsGen(CTimeStampsGen):
    
    def __init__(self,start:datetime.datetime,delta:datetime.timedelta,nLen):
        super().__init__(start,delta,nLen)
        
        
class CBitalinoWaveData(CWaveData): # EEG unit: uV; EOG unit: mv

    def __init__(self):
        super().__init__(-1,-1,CTimeStampsGen(0, 0, 1)) #still can't decide this param at this time for bitalino file
        
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
        
        for key in dataDescription.keys(): #now the key is just the mac address of the device
            dataDescription = dataDescription[key]
        
        self.timestamps = rowArrayNum[:,0]
        self.description = dataDescription
#        print(dateTime.datetime.now())
        if mode=='EEG':
            self.nChan = 1
            self.data = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-1],10,3.3,40000,'uV')), 0)
#            self.rawdata = np.expand_dims(rowArrayNum[:,-1],0)
            self.description["channelInfo"] = [[1],['EarEEG']]
        elif mode == 'EOG':
            self.nChan= 1
            self.data = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-2],10,3.3,2040, 'mV')), 0)
            self.description["channelInfo"] = [[1],['Eog']]
        elif mode == 'EEGandEOG':
            data1 = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-1],10,3.3,40000,'uV')), 0)
            data2 = np.expand_dims(np.array(self.getRealSignal(rowArrayNum[:,-2],10,3.3,2040, 'uV')), 0)
            self.nChan = 2
            self.data = np.concatenate([data1,data2],0)
            self.description['channelInfo'] = [[1,2],['EarEEG','Eog']]
        else:
            print("bitalino error: doesn't support this mode!")
#        print(dateTime.datetime.now())
        
        startTime = datetime.datetime.strptime( dataDescription['date'] + ' ' + dataDescription['time'], '%Y-%m-%d %H:%M:%S.%f')
        self.srate = dataDescription["sampling rate"]
        
        print("reading bitalinofile Finished")
        
        delta = datetime.timedelta(seconds = 1/self.srate)    
        self.timeStampsGen = CDateTimeStampsGen(startTime,delta,len(self.timestamps))#initiate the timestamp sequence generator
        self.calTimeStamp(self.timeStampsGen)
        
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
    
    def __len__(self):
        return len(self.data)
    