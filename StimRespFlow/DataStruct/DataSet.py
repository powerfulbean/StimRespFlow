# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:15:34 2019

@author: Jin Dou
"""

from .. import outsideLibInterfaces as outLib
from .. import DataIO
from .Abstract import CWaveData
from .LabelData import CLabelInfo, CLabels
from ..Helper.Protocol import CDataSetProtocol 
from enum import Enum, unique
import numpy as np
import warnings
from StellarInfra import siIO

# def fDummy(x):
#     return x

class CFlowDict(dict):
    
    # def __new__(cls,*args,**kwargs):
    #     obj = super().__new__(cls,*args,**kwargs)
    #     obj.fEncode = fDummy
    #     return obj
    
    def setInfo(self,inDict:dict):
        if self.__dict__.get('info'):
            self.info.update(inDict)
        else:
            self.info = dict(inDict)
            
    def fEncode(self,x):
        return x
    
    def toList(self):
        keyList = []
        valueList = []
        for key in self:
            keyList.append(key)
            valueList.append(self.__getitem__(key))
        return keyList,valueList
            
    def toDict(self):
        out = {}
        out['data'] = {}
        out['data'].update(self)
        out['info'] = self.__dict__.get('info')
        return out
        
   

@unique
class EOperation(Enum):
    Resample = 'resample'
    HighPass = 'highPass'
    LowPass = 'lowPass'
    BandPass = 'bandPass'
    Transform = "transform"

class CTrialOrganizor:
    '''
    Organize the data using key word of trials
    '''
    
    def __init__(self):
        pass
class CDataOrganizorLite:
    
    '''
    each of oData's timesamples indicates a data sample,
    each of oLabel's timesamples indicates a label related with a data sample in the oData,
    and there is one-to-one mapping between timestamps of oData and oLabel
    
    CDataOrganizorLite's DataRecord's data should be simple numpy array data
        its first dimension should indicate the channel's index of the data. 
        (index_chanel,sample)
    CDataOrganizorLite's DataRecord's stimuli should be simple numpy array data,
        its first dimension should indicate the class's index of the stimuli.
        (index_class,sample)
        
    there should be no time overlaps between its different DataRecords
    
    oLabel should only use CLabelInfoGeneral or its child class as the only type of CLabelInfo  
    '''
    
    def __init__(self,nChannels,srate,channelList,keyFunc=None):#keyFunc is used for timestamps comparison
        self.dataDict = dict() #key: (startTimeId, endTimeId); Value: CDataRecord
        self.nChannels = nChannels
        self.srate = srate
        self.channelList = channelList
        self.keyFunc = keyFunc
        self.oCheck = CDataSetProtocol()
    
    def insert(self,oData:CWaveData,oLabel:CLabels):
        if(oData.sampleRate != self.srate):
            raise ValueError()
        startId = oLabel.timestamps[0]
        endId = oLabel.timestamps[-1]
        for idx,timeId in enumerate(oLabel.timestamps):
            if(oData.timestamps[idx] != timeId):
                raise ValueError("timestamp of oData and oLabel doesn't match ", oData.timestamps[idx],timeId)
        
        data = oData.rawdata.copy()
        stimuli = oLabel.rawdata.copy()
        stimuliDes = oLabel.oLabelInfo.labelClassList.copy()
        self.dataDict[(startId,endId)] = CDataRecord(data,stimuli,stimuliDes,self.srate)
#        err = self.oCheck.check_DataOrganizorLite(self)
#        return err
        
    def getSortedKeys(self,reverse = False):
        keysList = list(self.dataDict.keys())
        ans = None
        if self.keyFunc == None:
            ans = sorted(keysList,key = None)
        else:
            ans = sorted(keysList,key = self.tupleKeyFunc)
        return ans
        
    def tupleKeyFunc(self,oTuple):
        firstKey = oTuple[0]
        if(self.keyFunc != None):
            return self.keyFunc(firstKey)
        else:
            return None
        
    def dataRecordBasedOnTime(self,reverse = False):
        keys = self.getSortedKeys(reverse)
        dataList = list()
        stimuliList = list()
        err = self.oCheck.check_DataOrganizorLite(self)
        if(err == False):
            raise ValueError("DataOrganizorLite's data doesn't obey relative CProtocol")
        for key in keys:
            dataList.append(self.dataDict[key].data)
            stimuliList.append(self.dataDict[key].stimuli)
        
        data = np.concatenate(dataList,axis = 1)
        stimuli = np.concatenate(stimuliList,axis = 1)
        stimuliDes = self.dataDict[keys[0]].stimuliDes
        srate = self.dataDict[keys[0]].srate
        
        ans_oDataRecord = CDataRecord(data,stimuli,stimuliDes,srate)
        return ans_oDataRecord
     
    def dataSetBasedOnStimuliDesc(self,LabelInfoClass,selectValue):
        self.oCheck.check_DataOrganizorLite(self)
        oDataset = CDataSet(LabelInfoClass)
        t = selectValue
        srate = self.srate
        for key in self.dataDict:
            eventsClass = self.dataDict[key].stimuliDes
            stimRow = eventsClass.index(LabelInfoClass)
            data = self.dataDict[key].data
            stimuli = self.dataDict[key].stimuli
            stimSelect = self.dataDict[key].stimuli[stimRow]
            idx = 0
            startIdx = 0
            endIdx = 0
            while(idx<len(stimSelect)-1):
                if(stimSelect[idx] != t and stimSelect[idx+1] == t ):
                    startIdx = idx +1
#                    print('start idx',startIdx)
                if(stimSelect[idx] == t and stimSelect[idx+1] != t ):
                    endIdx = idx + 1
#                    print('start and end idx',startIdx,endIdx)
                    dataTemp = data[:,startIdx:endIdx]
                    stimuliThis = stimuli[:,startIdx:endIdx]
                    oTempDatarecord = CDataRecord(dataTemp,stimuliThis,eventsClass,srate)
                    oDataset.dataRecordList.append(oTempDatarecord)
                    startIdx = 0
                    endIdx = 0
                idx += 1
        eventList = len(oDataset.dataRecordList) * [t]
        
        return oDataset,eventList
    
    def __getitem__(self,Key):
        return self.dataDict[Key]
            
            
            
    
class CDataOrganizor:
    '''
    !Important: the labels (CLabel) in this class have the same memory addresses 
    as the labels it adds by the "addLabels" function 
    '''
    
    '''
    organize data for a single CLabel object
    '''
    
    def __init__(self,n_channels,srate, channelsList):
        self.labels = dict() # key: DataStruct.LabelData.CLabelInfo, value: data
        self.labelList = list()
        self.type = '' #was allocated value when build DataOrganizer from '.mat' files which was built by saveToMat
        self.srate = srate
        self.n_channels = n_channels
        self.channelsList = channelsList
        self.outLibIOObject = outLib._OutsideLibIO()
        self._opLogs = list()
        
    def logOp(self,chnnName:str, op:EOperation, params:list):
        record = (chnnName, op.value,params)
        self._opLogs.append(record)
    
    def addLabels(self,oLabel:CLabels):
        inputLabels = oLabel.timestamps
        self.type = oLabel.type
        for idx,i in enumerate(inputLabels):
            stimuli = oLabel.rawdata[idx]
            self.labels[i] = CDataRecord(None,stimuli,[{'name':stimuli.name,'otherNames':stimuli.otherNames}],self.srate)
        self.labelList = list(self.labels.keys())
    
    def __getitem__(self,label:CLabelInfo):
        return self.labels[label]
    
    def __setitem__(self, label:CLabelInfo , value):
        self.labels[label] = value
    
#    def buildLabelslist(self):
#        self.labelList = [i for i in self.labels]
#        self.labelList.sort(key=self.labelList[0].sorted_key)
    
    def checkDataSet(self):
        if(len(self.data) == len (self.labels)):
            return True
        else:
            return False
        
    def saveToMat(self,OutputFolder):
        
        DataIO.checkFolder(OutputFolder)
        for i in self.labels:           
            #Label Part
            Output = dict()
            MAT = dict()
            MAT['LabelName'] = i.name
            MAT['OtherLabelNames'] = i.otherNames
            if(type(i.data) == str):
                pass
            else:
                MAT['Label'] = i.data.mainStream
                MAT['OtherLabel'] = i.data.otherStreams
            MAT['ChannelList'] = self.channelsList #will be formated and aligned in 3 columns char vector in Matlab
            MAT['Type'] = i.type #save the type of the label as the type of the organizer
            MAT['SamplingRate'] = self.srate
            MAT['Time'] = [str(i.startTime).replace(":","-"),str(i.endTime).replace(":","-")]
            
            #Data Part
            MAT['Data'] = self.labels[i]
            
            Output['OriDataSeg'] = MAT
            fileName = '['+ MAT['Type'] +']'+ MAT['Time'][0] + '_' + MAT['Time'][1] + '_' + MAT['LabelName'] + '.mat'
            fileName.replace(" ","_")
            self.outLibIOObject.writeMATFile(Output,OutputFolder + fileName)
    
    def readFromMAT(self,files):
        datetimeMod = outLib._OutsideLibTime()._importDatetime()
        for idx, file in enumerate(files):
            Dict = outLib._OutsideLibIO().loadMATFile(file)
            if(idx == 0):#read from the firts file (which represents all the files)
                self.channelsList = [str(i) for i in Dict["ChannelList"] ]
                self.type = str(Dict["Type"])
                self.srate = Dict["SamplingRate"][0][0]
                self.n_channels = len(self.channelsList)
            startTime = datetimeMod.datetime.strptime(str(Dict["Time"][0]), '%Y-%m-%d %H-%M-%S.%f')
            endTime = datetimeMod.datetime.strptime(str(Dict["Time"][1]), '%Y-%m-%d %H-%M-%S.%f')
            label = CLabelInfo(str(Dict["LabelName"][0]),idx,startTime,endTime)
            if (len(Dict["OtherLabelNames"]) != 0):
                label.otherNames.append(str(Dict["OtherLabelNames"][0]))
            self.labels[label] = Dict["Data"]
    
    def assignTargetData(self,targetList,frontLag_s = 0, postLag_s = 0): #need to be improved
        ''' 
        Des:
            assign raw data to self.labels(CLabels) according to the absolute time of raw data and labels
        
        Params:
            targetList: list of CRawData objects
            frontLag_s: time included ahead of the label's start time, the unit is secound
            postLag_s: time included after of the label's end time, the unit is secound
        '''
        index = 0
        remove_list = list()
        breakFlag = False
        for label in self.labels:
            while(True):
                breakFlag = False
                dataObject = targetList[index]
                [data,interval] =  dataObject.findInterval(label.startTime,label.endTime,frontLag_s,postLag_s)
                index_hist = index
                if (interval == False):## if this DataFile doesn't contain any required data, jump to next file
                    while(interval == False):
                        index+=1
                        if(index == len(targetList)):
                            print("total data lost: data between: " + str(label.startTime) + ' and ' + str(label.endTime) + " is not found in the data file")
                            remove_list.append(label)
                            index = index_hist
                            breakFlag = True
                            break
                        else:
                            dataObject = targetList[index]
                            [data,interval ] =  dataObject.findInterval(label.startTime,label.endTime,frontLag_s,postLag_s)
                elif (interval[0] <= label.startTime and interval[1] >= label.endTime ):
                    self.labels[label].data = data
#                    print(index,' ',interval[0],interval[1],' for ',label.startTime,label.endTime)
        #            print(interval)
                    break
                else:## if this DataFile just contains part of the required data, jump to next file, and combine the data
                    index+=1
                    dataObject = targetList[index]
                    [data1,interval1 ] =  dataObject.findInterval(label.startTime,label.endTime,frontLag_s,postLag_s)
        #            print(index)
                    if(interval1 == False):
#                        if(interval == False):
#                            print("total data lost: data between: " + str(label.startTime) + ' and ' + str(label.endTime) + " is not found in the data file")
#                            remove_list.append(label)
#                        else:
                        print("part of the data lost: data between: " + str(interval[1]) + ' and ' + str(label.endTime) + " is not found in the data file")
                        self.labels[label].data = data
                        break
                    else:
                        data = np.concatenate([data,data1],axis=1)
                        interval[1] = interval1[1]
                
                if(breakFlag == True):
                    break
                
        for to_remove in remove_list:
            self.labels.pop(to_remove,None)
    
    def getEpochDataSet(self,epochLen_s):
#        transObject = outLib._OutsideLibTransform()
        oDataSet = CDataSet(self.type + str(self.labelList[0].startTime))
        for label in self.labels:
            data = self.labels[label].data
            stimuli = self.labels[label].stimuli # label stores a related markerRecord
            # get segmented auditoryStimuli Object from the markerRecord's whole length auditoryStimuli Object ('data' attribute) 
            Num = round(label.getDuration_s() / epochLen_s)
            ans = stimuli.getSegmentStimuliLists(epochLen_s,Num) 
            labelDes = [stimuli.name, stimuli.otherNames[0]]
#            print('Num: ',Num)
            for i in range(Num):
#                print(data.shape,i*epochLen_s*self.srate,(i+1)*epochLen_s*self.srate)
#                print((i+1)*epochLen_s*self.srate)
                dataSeg = data[:,int(i*epochLen_s*self.srate) : int((i+1)*epochLen_s*self.srate)]
                label = ans[i]
                srate = self.srate
                recordTemp = CDataRecord(dataSeg,label,labelDes,srate)
                if(i*epochLen_s*self.srate <= data.shape[1]):
#                    print('append',i*epochLen_s*self.srate,data.shape[1])
                    recordTemp.filterLog = self._opLogs.copy()
                    oDataSet.dataRecordList.append(recordTemp)
        
        return oDataSet
    

class CDataSet:
    def __init__(self,dataSetName = None):
        self.dataRecordList = list()
        self.name = dataSetName
        self.srate = -1
        self.desc = dict()
        self.stimuliDict = {}
        self.recordDict = {}
    
    @property
    def records(self,):
        return self.dataRecordList
    
    def append(self,item):
        self.records.append(item)
    
    def __getitem__(self,idx):
        return self.records[idx]
    
    def __len__(self,):
        return len(self.records)
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n  < len(self.records):
            self.n += 1
            return self.records[self.n-1]
        else:
            raise StopIteration
        
    def selectByInfo(self,keyWord:list):
        output = list()
        for record in self:
            if all([any([key in info for info in record.descInfo]) for key in keyWord]):
                output.append(record)
        return output
        
    def constructFromFile(self,fileName):
        import pickle
        file = open(fileName, 'rb')
        temp = pickle.load(file)
        for key in temp.__dict__:
            setattr(self,key,getattr(temp,key))
    
    def save(self,folderName,name = None):
        DataIO.checkFolder(folderName)
#        print(self.name)
        if(name == None):
            file = open(folderName + '/' + self.name.replace(':','_') + '.bin', 'wb')
        else:
            file = open(folderName + '/' + name + '.bin', 'wb')
        import pickle
        pickle.dump(self,file)
        file.close()
        
    def sort(self,keyFunc):
        self.dataRecordList.sort(key = keyFunc)
        return self
    
    def clear(self):
        self.dataRecordList.clear()
            
    def clearRef(self,indices):
        assert type(indices) == list
        for index in sorted(indices, reverse=True):
            self.dataRecordList[index] = None
    
    # def __del__(self):
    #     print(f"{self.__class__} dataRecordsList clear")
    #     self.dataRecordList.clear()
        
    def __add__(self,dataset2):
        if len(set(self.stimuliDict.keys()) & set(dataset2.stimuliDict.keys())) != 0:
            warnings.warn('there are overlaps between the stimuliDict of two datasets')
        newDataset = CDataSet()
        if len(self.records) == 0:
            newDataset.name = dataset2.name
            newDataset.desc = dataset2.desc
            newDataset.srate = dataset2.srate
        else:
            newDataset.name = self.name
            newDataset.desc = self.desc
            newDataset.srate = self.srate
        newDataset.dataRecordList += self.dataRecordList
        newDataset.dataRecordList += dataset2.dataRecordList
        newDataset.stimuliDict.update(self.stimuliDict)
        newDataset.stimuliDict.update(dataset2.stimuliDict)
        return newDataset
    
    
    def subset(self,indices):
        newDataset = CDataSet()
        newDataset.name = self.name
        newDataset.desc = self.desc
        newDataset.srate = self.srate
        newDataset.dataRecordList = [self.dataRecordList[idx] for idx in indices]
        newDataset.stimuliDict.update(self.stimuliDict)
        return newDataset
    
    def buildDict(self):
        #build record dict using descInfo of each record
        if getattr(self,'recordDict',None) is None:
            self.recordDict = {}
        for i in self:
            self.recordDict['_'.join(i.descInfo)] = i
        
    @classmethod
    def fromMatCells(cls,fileName):
        newDataset = cls(fileName)
        mat = siIO.loadMatFile(fileName)
        stims = np.squeeze(mat['stim'])
        fs = np.squeeze(mat['fs'])
        resps = np.squeeze(mat['resp'])
        newDataset.srate = fs
        for idx,_ in enumerate(stims):
            record = CDataRecord(resps[idx], stims[idx], str(idx), fs)
            newDataset.append(record)
        return newDataset
        
        
class CDataRecord: #base class for data with label
    def __init__(self,data,stimuli,stimuliDes:list,srate):
        self.data = data #real data Segment
        self.stimuli = stimuli #segmented "auditoryStimuli" object
        self.stimuliDes = stimuliDes #information of the stimuli
        self.srate = srate
        self.filterLog = list()
        self.descInfo = stimuliDes
        
    def errorPrint(self,error):
        print("CDataRecorder error: " + error)
        
class CDataDict:
    ''' assume the first dimension is channel'''
    def __init__(self,dataDict:dict):
        self._data = dataDict
        self.keys = list(self._data.keys())
        self.defaultKeySeq:list = list()
        self.channelAxis = 0
        for key in self._data:
            setattr(self,key,self._data[key])
        
    @property    
    def T(self,):
        Dict = dict()
        for i in self._data:
            Dict[i] = self._data[i].T
        return CDataDict(Dict)
    
    @property
    def data(self,):
        return self.arrayCat(self.defaultKeySeq)
    
    @data.setter    
    def data(self,x):
        self._data = x
    
    def arrayCat(self,keySeq):
        oList = list()
        for i in keySeq:
            oList.append(self._data[i])
        return np.concatenate(oList,axis = self.channelAxis)
      
class CDataDictRecord(CDataRecord,CDataDict):
    
    def __init__(self,data:dict,stimuli,stimuliDes:list,srate):
        CDataDict.__init__(self,data)
        CDataRecord.__init__(self,data,stimuli,stimuliDes,srate)
        
        
    
        
    


        
            
        