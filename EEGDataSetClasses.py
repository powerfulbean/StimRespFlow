# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:26:53 2019

@author: sam jin dou
"""

import outsideLibInterfaces as outLib
import os
import numpy as np
import DataIO
from preprocessing import CPreprocess
from AuditoryAttentionApplication.MNEHelper import CStimuliCache
from datetime import time as TimeObject

class RawData:
    
    def __init__(self): #the length of timeStamps should be as same as the number of samples 
        self.rawdata = list()
        self.timestamps = list() 
        self.startTime = '' # store the datatime.datetime object
        self.description = dict()
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
        return the interval that satisfy the requirement most
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
        
        
class bitalinoRawdata(RawData): # EEG unit: uV; EOG unit: mv
        
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
        
        #return data, dataDescription
    
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
    
#    def caliTime(self):
#        pass
        
class gtecRawdata(RawData): # EEG unit: uV
    def readFile(self,fileNameDes):
        import json
        self.description = json.loads(DataIO.loadText(fileNameDes))
        fileNameData = self.description['DataName']
        self.sampleRate = self.description['frequency']
        dateTime = outLib._OutsideLibTime()._importDatetime()
        self.startTime = dateTime.datetime.strptime(self.description['StartDate'],'%Y-%m-%dT%H:%M:%S.%fZ')
        data = DataIO.loadBinFast(fileNameData)
        num_channels = self.description['numChannels']
        self.rawdata = np.reshape(data,(num_channels,-1))
        self.timestamps = [0] * self.rawdata.shape[1]
        self.numChannels = self.description['numChannels']
     

class Labels(RawData): # for labels created by psychopy, the last number is microsecond but not millisecond              
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
        if not os.path.isdir(folder):
            os.mkdir(folder)
        fileAddress = folder + self.description+"_labels_" + str(self.startTime).replace(":","-") + ".jin" 
        f = open(fileAddress,'w+')
        f.write(app_json)
        f.close
    
    def stimuliEventParser(self,buffer, stimuliStartTag, StimuliEndTag, i, recordObject):
        stimuliStart = buffer[i]
        if(stimuliStart[0] == stimuliStartTag):
            temp = stimuliStart[1]
            realName = temp
            recordObject.name = realName
            datetime = outLib._OutsideLibTime()._importDatetime() 
            
            temp = stimuliStart[2:len(stimuliStart)] # calculate start time
            [h,m,s,ms] = [int(t) for t in temp]
            startTime = datetime.time(h,m,s,ms)
            recordObject.startTime = startTime
            i+=1
            
            stimuliEnd = buffer[i]
            #the audio End Label
            if(stimuliEnd[0] == StimuliEndTag):
                recordObject.index = stimuliEnd[1]
                temp = stimuliEnd[2:len(stimuliEnd)] # calculate end time
                [h,m,s,ms] = [int(t) for t in temp]
                endTime = datetime.time(h,m,s,ms)
                recordObject.endTime = endTime
                i+=1
            else:
                print("visuallabels, parser error")
        else:
            print("visuallabels, parser error")
        
        return i, recordObject

    def writeType(self,typeName):
        for i in self.timestamps:
            i.type = typeName
            
    def promoteMarker(self):
        '''
        change the datetime.time (doesn't contain information of year,month,day) object into datetime.datetime object
        '''
        datetime = outLib._OutsideLibTime()._importDatetime()
        if(len(self.timestamps) != len(self.rawdata)):
            print("label warning: the number of rawdata is wrong")#, length of raw data is: ", len(self.rawdata))
        for i,marker in enumerate(self.timestamps):
            if (i<len(self.rawdata)):
                marker.promote(datetime,self.startTime.date(),self.rawdata[i])
            else:
                marker.promoteTime(datetime,self.startTime.date())

class BlinksCaliLabels(Labels):
    
    def readFileForInfo(self, fileName):
        self.outLibIO = outLib._OutsideLibIO()
        tempName,t = os.path.splitext(os.path.basename(fileName))
        self.description = 'BlinksCali_' + tempName
        buffer = []
        with open(fileName, 'r') as the_file: #opoen the labels file
            lines = the_file.readlines()
        
        for line in lines: # read the label file
            temp = line.split()
            buffer.append(temp)
            
        i = 0
        self.startTime = self.parseTimeString(buffer[i][1] + ' ' + buffer[i][2])
        while( i < len(buffer)): #buffer is the whole document
            tempRecord = markerRecord('','','','') #store crossing time
            #the first line, save the start date
            Type = buffer[i][0]
            if(Type == 'blink' or Type == 'lookLeft' or Type == 'lookRight'):
                tempRecord.type = 'cali'
                if(buffer[i][0] == 'blink'):
                    tempRecord.name = 'blink'
                elif(buffer[i][0] == 'lookLeft'):
                    tempRecord.name = 'lookLeft'
                elif(buffer[i][0] == 'lookRight'):
                    tempRecord.name = 'lookRight'
                time = self.parseTimeString(buffer[i][1] + ' ' + buffer[i][2]).time() #read as datetime and change to time
                tempRecord.startTime = time
                tempRecord.endTime = time
                tempRecord.type = 'cali'
                self.timestamps.append(tempRecord)
                self.rawdata.append(self.description + '_' + Type)
                i += 1
            else:
                print("BlinksCaliLabels, readFile error")
                return
        
    def parseTimeString(self,string):
        dateTime = outLib._OutsideLibTime()._importDatetime() 
        return dateTime.datetime.strptime(string , '%Y-%m-%d %H:%M:%S.%f')
    
class AuditoryLabels(Labels):       
    
    def readFile(self,fileName,stimuliDir): #can't work for label files with 'Single' type 
        self.outLibIO = outLib._OutsideLibIO()
        self.description = 'Auditory'
        buffer = []
        datetime = outLib._OutsideLibTime()._importDatetime()    
        
        with open(fileName, 'r') as the_file: #opoen the labels file
            lines = the_file.readlines()
        
        for line in lines: # read the label file
            temp = line.split()
            buffer.append(temp)
#        print(len(buffer))
        i = 0
        while( i < len(buffer)): #buffer is the whole document
#            print("i="+str(i))
            if(i == 0):
                [a,b,c,d,e,f] = [int(j) for j in buffer[i]]
                self.startTime = datetime.datetime(a,b,c,d,e,f*1000) # Lib: datetime
                i+=1
                
            if(buffer[i][0]=='left' or buffer[i][0]=='right'):
                
                leftflag = ''
                if(buffer[i][0] == 'left'):
                    leftflag = True
                else:
                    leftflag = False
                
                tempRecord = markerRecord('','','','')
                
                i += 1
                audioStart = buffer[i]
                
                if(audioStart[0] == 'audioStart'):
                    stimuliObject = auditoryStimuli()
                    temp = audioStart[1] # real audio name
                    realName, extension = os.path.splitext(temp)
                    
                    stimuliName = self.stimuliname(leftflag, realName)# main stimuli name
                    tempRecord.name = stimuliName
                    #print(stimuliName)
                    stimuli = self.readStimuli(stimuliDir+stimuliName+extension)# stimuli 
                    stimuliObject.mainStream = stimuli
                    
                    otherstimulisName = self.stimuliname(not leftflag, realName)# other stimuli name
                    tempRecord.otherNames.append(otherstimulisName)
                    #print(otherstimulisName)
                    otherStimulis = self.readStimuli(stimuliDir+otherstimulisName+extension)#other stimulis
                    stimuliObject.otherStreams.append(otherStimulis)
                    
                    self.rawdata.append(stimuliObject)
                    temp = audioStart[2:len(audioStart)]
                    [h,m,s,ms] = [int(t) for t in temp]
                    startTime = datetime.time(h,m,s,ms)
                    tempRecord.startTime = startTime
                    
                    i+=1
                    audioEnd = buffer[i]
                    
                    if(audioEnd[0] == 'audioEnd'):
                        tempRecord.index = audioEnd[1]
                        temp = audioEnd[2:len(audioEnd)]
                        [h,m,s,ms] = [int(t) for t in temp]
                        endTime = datetime.time(h,m,s,ms)
                        tempRecord.endTime = endTime
                
                tempRecord.type = 'auditory'
                self.timestamps.append(tempRecord)
                i+=1
            elif(buffer[i][0]=='-1'):
                break
            else:
                print("labels, loadfile error")
        
#        self.writeType("auditory")    
        return buffer
    
    
    def readFileForInfo(self,fileName):
        
        #include the OutsideLib module for Input/Output
        self.outLibIO = outLib._OutsideLibIO()
        self.description = 'Auditory'
        buffer = []
        datetime = outLib._OutsideLibTime()._importDatetime()    
        
        with open(fileName, 'r') as the_file: #opoen the labels file
            lines = the_file.readlines()
        
        for line in lines: # read the label file
            temp = line.split()
            buffer.append(temp)
#        print(len(buffer))
        i = 0
        while( i < len(buffer)): #buffer is the whole document
#            print("i="+str(i))
#            print(buffer[i][0])
            
            #the first line, save the start date
            if(i == 0):
                [a,b,c,d,e,f] = [int(j) for j in buffer[i] if (j != '-' and j != ':') ]
                self.startTime = datetime.datetime(a,b,c,d,e,f) # Lib: datetime
                i+=1
            
            #the left/right label
            if(buffer[i][0]=='left' or buffer[i][0]=='right' or buffer[i][0] == 'Single'):
                
                leftflag = ''
                singleflag = False
                if(buffer[i][0] == 'left'):
                    leftflag = True
                elif(buffer[i][0] == 'Single'):
                    singleflag = True
                else:
                    leftflag = False
                
                tempRecord = markerRecord('','','','')
                
                i += 1
                audioStart = buffer[i]
                
                #the audio Start Label
                if(audioStart[0] == 'audioStart'):
                    # real audio name    
                    temp = audioStart[1] 
                    realName, extension = os.path.splitext(temp)
                    
                    if(singleflag == False):
                        # stimuli name
                        stimuliName = self.stimuliname(leftflag, realName)
                        tempRecord.name = stimuliName
                        #print(stimuliName)
                        otherstimulisName = self.stimuliname(not leftflag, realName)# other stimuli name
                        tempRecord.otherNames.append(otherstimulisName)
                        #print(otherstimulisName)
                    else:
                        tempRecord.name = realName
                        tempRecord.otherNames.append(realName)
                    
                    # calculate start time
                    temp = audioStart[2:len(audioStart)] 
                    [h,m,s,ms] = [int(t) for t in temp]
                    startTime = datetime.time(h,m,s,ms)
                    tempRecord.startTime = startTime
                    i+=1
                    
                    #the audio End Label
                    audioEnd = buffer[i]
                    if(audioEnd[0] == 'audioEnd'):
                        tempRecord.index = audioEnd[1]
                        temp = audioEnd[2:len(audioEnd)]
                        [h,m,s,ms] = [int(t) for t in temp]
                        endTime = datetime.time(h,m,s,ms)
                        tempRecord.endTime = endTime
                
                tempRecord.type = 'auditory'
                self.timestamps.append(tempRecord)
                i+=1
            elif(buffer[i][0]=='-1'):
                break
            else:
                raise TypeError("labels, loadfile error", buffer[i][0])
        
        self.writeType("auditory")              
        return buffer        
            
    
    def readStimuli(self,file):
        
        print("AuditoryLabels, readStimuli:", file)
        frameNum, channelNum, data = self.outLibIO.getMonoChannelData(file)
        a = DataIO.unpackWav(frameNum,channelNum,data)
        
        return np.asarray(a)
    
    def stimuliname(self,isLeft, realName):
        idx = realName.find('R')
        if(isLeft == True):
            realStimuli = realName[1:idx] # don't include the 'L'
        else:
            realStimuli = realName[idx+1:len(realName)] # don't include the 'R'
        return realStimuli

class VisualLabels(Labels):
    
    def readFileForInfo(self,fileName, readStimuli = False, stimuliDir = None):
        ''' 
        read all the necessary information from the label files
        
        '''
        #include the OutsideLib module for Input/Output
        self.outLibIO = outLib._OutsideLibIO()
        self.description = 'Visual'
        buffer = []
        datetime = outLib._OutsideLibTime()._importDatetime()    
        
        with open(fileName, 'r') as the_file: #opoen the labels file
            lines = the_file.readlines()
        
        for line in lines: # read the label file
            temp = line.split()
            buffer.append(temp)
            
        i = 0
        while( i < len(buffer)): #buffer is the whole document
            #the first line, save the start date
            if(i == 0):
                [a,b,c,d,e,f] = [int(j) for j in buffer[i] if (j != '-' and j != ':') ]
                self.startTime = datetime.datetime(a,b,c,d,e,f) # Lib: datetime
                i+=1
                
            if(buffer[i][0] == 'First' or buffer[i][0] == 'Second' or buffer[i][0] == 'Cat' ):
                stimuliIdx = 0
                if(buffer[i][0] == 'First'):
                    stimuliIdx = 1
                elif( buffer[i][0] == 'Second'):
                    stimuliIdx = 2
                elif(buffer[i][0] == 'Cat'):
                    stimuliIdx = 0
                else:
                    stimuliIdx = -1
                i+=1
                                 
                while(buffer[i][0] == 'sub'):
                    tempRecord = markerRecord('','','','') #store crossing time
                    tempRecord2 = markerRecord('','','','') #store stimuli time
                    tempRecord3 = markerRecord('','','','') #store rest time
                    i+=1
                    
                    if(buffer[i][0] == 'attendStart'):
                        i,tempRecord = self.stimuliEventParser(buffer,'attendStart','attendEnd',i,tempRecord)
                    elif(buffer[i][0] == 'unattendStart'):
                        i,tempRecord = self.stimuliEventParser(buffer,'unattendStart','unattendEnd',i,tempRecord)
                    elif(buffer[i][0] == 'crossingStart'):
                        i,tempRecord = self.stimuliEventParser(buffer,'crossingStart','crossingEnd',i,tempRecord)
                    else:
                        raise Exception("don't recognize the type "+ str(buffer[i][0])+' please check the file')
                    i,tempRecord2 = self.stimuliEventParser(buffer,'imageStart','imageEnd',i,tempRecord2)
                    i,tempRecord3 = self.stimuliEventParser(buffer,'restStart','restEnd',i,tempRecord3)
                    
                    tempName,t = os.path.splitext(tempRecord2.name)
                    tempRecord3.name = tempName + '_' + 'imagination' 
                    tempRecord.type = 'cross'
                    tempRecord2.type = 'image'
                    tempRecord3.type = 'rest'
                
                    self.timestamps.append(tempRecord)
                    self.rawdata.append('crossing')
                    self.timestamps.append(tempRecord2)
                    self.rawdata.append(str(stimuliIdx))
                    self.timestamps.append(tempRecord3)
                    self.rawdata.append('rest')
                    
            elif(buffer[i][0]=='-1'):
                break
            
            else:
                print("visuallabels, readFile error")
                return
            
#        self.writeType("visual")  
  
class AttentionLabels(Labels):
    
    def readFileForInfo(self,fileName, readStimuli = False, stimuliDir = None):
        self.outLibIO = outLib._OutsideLibIO()
        self.description = 'Attention'
        buffer = []
        datetime = outLib._OutsideLibTime()._importDatetime()    
        
        with open(fileName, 'r') as the_file: #opoen the labels file
            lines = the_file.readlines()
        
        for line in lines: # read the label file
            temp = line.split()
            buffer.append(temp)
            
        i = 0
        while( i < len(buffer)): #buffer is the whole document
            #the first line, save the start date
            if(i == 0):
                [a,b,c,d,e,f] = [int(j) for j in buffer[i] if (j != '-' and j != ':') ]
                self.startTime = datetime.datetime(a,b,c,d,e,f) # Lib: datetime
                i+=1
                
            if(buffer[i][0] == 'P' or buffer[i][0] == 'A' or buffer[i][0] ==  'R'):
                stimuliIdx = 0
                if(buffer[i][0] == 'P'):
                    stimuliIdx = 'P'
                elif( buffer[i][0] == 'A'):
                    stimuliIdx = 'A'
                elif(buffer[i][0] == 'R'):
                    stimuliIdx = 'R'
                else:
                    stimuliIdx = -1
                i+=1
                                 
                while(buffer[i][0] == 'taskStart'):
                    tempRecord = markerRecord('','','','') #store crossing time
                    tempRecord.type = 'attention'
                    
                    i,tempRecord = self.stimuliEventParser(buffer,'taskStart','taskEnd',i,tempRecord)
                    tempRecord.name = stimuliIdx + tempRecord.name
                    
                    self.timestamps.append(tempRecord)
                    self.rawdata.append(str(stimuliIdx))
                
                if(buffer[i][0] == 'restStart'):
                    tempRecord = markerRecord('','','','') #store crossing time
                    tempRecord.type = 'attention'
                    
                    i,tempRecord = self.stimuliEventParser(buffer,'restStart','restEnd',i,tempRecord)
                    tempRecord.name = 'Break' + tempRecord.name
                    
                    self.timestamps.append(tempRecord)
                    self.rawdata.append('Break')
                    
            elif(buffer[i][0]=='-1'):
                break
            
            else:
                print("visuallabels, readFile error")
                return
        
        #self.writeType("attention")
        

class auditoryStimuli:
    def __init__(self):
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
            tempStimuli = auditoryStimuli()
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
        tempStimuli = auditoryStimuli()
        tempStimuli.mainStream = self.mainStream[startIdx:endIdx]
        tempStimuli.otherStreams.append(self.otherStreams[0][startIdx:endIdx])
        tempStimuli.len_s = int(endTime- startTime)
        tempStimuli.otherLen_s.append(int(endTime - startTime))
#        print('len2',len(tempStimuli.mainStream))
        return tempStimuli
        

            
class markerRecord:
    
    def __init__(self,name,index,startTime,endTime):
        self.name = name
        self.otherNames = list() # background stimuli name
        self.index = index
        self.startTime = startTime #datetime.time or datetime.datetime object
        self.endTime = endTime #dateime.time or datetime.datetime object 
        self.type = ''
        self._promoteFlag = False
        self._promoteTimeFlag = False
        self.data = '' #label info (such as auditoryStimuli object)
        
    def getLabelDict(self,):
        if (self.type == 'attention') :
            name = self.name 
        elif (self.type == 'image' or self.type == 'rest' or self.type == 'cross'):
            name = self.name
        elif (self.type == 'cali'):
            name = self.name
        elif (self.type == 'auditory'):
            name = self.name + '_n_' + self.otherNames[0]
        else:
            name = self.name[self.name.find('-')+1:len(self.name)]
            name = self.name[0:self.name.find('_')]
        dict1 = {
                'name':name+'_Start',
                'time':str(self.startTime),
                }
        
        dict2 = {
                'name':name+'_End',
                'time':str(self.endTime),
                }
        return dict1,dict2
    
    def checkPromoto(self,):
        return self._promoteFlag
    
    def promote(self,datetimeModule,dateObject, data):
        if(self._promoteFlag == True):
            print("marker has been promoted")
        else:
            self.promoteTime(datetimeModule,dateObject)    
            self.data = data
            self._promoteFlag = True
    
    def promoteTime(self,datetimeModule,dateObject):
        if(self._promoteTimeFlag == True):
            print("time of the marker has been promoted")
        else:
            startTime = datetimeModule.datetime(1,1,1,1,1,1)
            endTime = datetimeModule.datetime(1,1,1,1,1,1)
            self.startTime = startTime.combine(dateObject, self.startTime)
            self.endTime = endTime.combine(dateObject, self.endTime)
            self._promoteTimeFlag = True
    
    def sorted_key(self,x):
        return x.startTime

class DataOrganizor:
    
    def __init__(self,n_channels,srate, channelsList):
        self._data = list()
        self.labels = dict() # key: CLabelInfo, value: the actual data
        self.labelList = list()
        self.type = '' #was allocated value when build DataOrganizer from '.mat' files which was built by saveToMat
        self.srate = srate
        self.n_channels = n_channels
        self.channelsList =channelsList
        self.outLibIOObject = outLib._OutsideLibIO()
        self.preprocessObject = CPreprocess()
    
    def addLabels(self,inputLabels):
        for i in inputLabels:
            self.labels[i] = ''
    
    def buildLabelslist(self):
        self.labelList = [i for i in self.labels]
        self.labelList.sort(key=self.labelList[0].sorted_key)
    
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
            label = markerRecord(str(Dict["LabelName"][0]),idx,startTime,endTime)
            if (len(Dict["OtherLabelNames"]) != 0):
                label.otherNames.append(str(Dict["OtherLabelNames"][0]))
            self.labels[label] = Dict["Data"]
            
    def loadStimulis(self,Folder, extension, oCache : CStimuliCache = None):
        if(extension == '.wav'):
            for label in self.labels:
                mainStreamName = label.name
                otherStreamNames = label.otherNames
                print("AuditoryLabels, readStimuli:", mainStreamName,otherStreamNames[0])
                stimuliObject = auditoryStimuli()
                mainStream, len_s = DataIO.readStimuli(Folder + mainStreamName + extension) 
                stimuliObject.mainStream = mainStream
                stimuliObject.len_s = len_s
                for otherStreamName in otherStreamNames:
                    otherStream, otherLen_s = DataIO.readStimuli(Folder + otherStreamName + extension)
                    stimuliObject.otherStreams.append(otherStream)
                    stimuliObject.otherLen_s.append(otherLen_s)
                    
                # save this auditoryStimuli object to the data attribute of this markerRecord object
                label.data = stimuliObject
        
        elif(extension == 'cache'):
            for label in self.labels:
                mainStreamName = label.name
                otherStreamNames = label.otherNames
                print("AuditoryLabels, read Stimuli from cache:", mainStreamName,otherStreamNames[0])
                stimuliObject = auditoryStimuli()
                mainStream, len_s = oCache.getStimuli(mainStreamName)
                stimuliObject.mainStream = mainStream
                stimuliObject.len_s = len_s
                for otherStreamName in otherStreamNames:
                    otherStream, otherLen_s = oCache.getStimuli(otherStreamName)
                    stimuliObject.otherStreams.append(otherStream)
                    stimuliObject.otherLen_s.append(otherLen_s)
                    
                # save this auditoryStimuli object to the data attribute of this markerRecord object
                label.data = stimuliObject
            
    def getEpochDataSet(self,epochLen_s,stimuliMode = 'StimuliBase'):
#        transObject = outLib._OutsideLibTransform()
        if(stimuliMode == 'StimuliBase'):
            dataSetObject = CDataSet(self.type + str(self.labelList[0].startTime))
            for label in self.labels:
                data = self.labels[label]
                stimuli = label # label stores a related markerRecord
                # get segmented auditoryStimuli Object from the markerRecord's whole length auditoryStimuli Object ('data' attribute) 
                ans, Num = stimuli.data.getSegmentStimuliLists(epochLen_s) 
                labelDes = [stimuli.name, stimuli.otherNames[0]]
                print('Num: ',Num)
                for i in range(Num):
                    recordTemp = CDataRecord()
    #                print(data.shape,i*epochLen_s*self.srate,(i+1)*epochLen_s*self.srate)
    #                print((i+1)*epochLen_s*self.srate)
                    recordTemp.dataSeg = data[:,int(i*epochLen_s*self.srate) : int((i+1)*epochLen_s*self.srate)]
                    recordTemp.label = ans[i]
                    recordTemp.srate = self.srate
                    recordTemp.labelDes = labelDes
                    if(i*epochLen_s*self.srate<data.shape[1]):
    #                    print('append',data.shape[1])
                        dataSetObject.dataRecordList.append(recordTemp)
        
        elif(stimuliMode == 'DataIntvl'):
            '''
            only for online model it can be used when the start time and end time of the label is allocated in Adaptor.MNEEvokedToDataOrganizor function
            '''
            dataSetObject = CDataSet(self.type + str(self.labelList[0].startTime))
            for label in self.labels:
                data = self.labels[label]
                startTime = label.startTime
                endTime = label.endTime
                stimuli = label # label stores a related markerRecord
                # get segmented auditoryStimuli Object from the markerRecord's whole length auditoryStimuli Object ('data' attribute) 
                ans = stimuli.data.getSegmentByTime(startTime,endTime) 
                labelDes = [stimuli.name, stimuli.otherNames[0]]
                recordTemp = CDataRecord()
#                print(data.shape,i*epochLen_s*self.srate,(i+1)*epochLen_s*self.srate)
#                print((i+1)*epochLen_s*self.srate)
                recordTemp.dataSeg = data
                recordTemp.label = ans
                recordTemp.srate = self.srate
                recordTemp.labelDes = labelDes
                dataSetObject.dataRecordList.append(recordTemp)
        
        else:
            raise ValueError('invalide mode')
        
                
        return dataSetObject
    
    def getEpochDataSetForVisual(self,epochLen_s):
        dataSetObject = CDataSet(self.type + str(self.labelList[0].startTime))
        for label in self.labels:
            data = self.labels[label]
            stimuli = label # label stores a related markerRecord
            # get segmented auditoryStimuli Object from the markerRecord's whole length auditoryStimuli Object 
            labelDes = [stimuli.name]
            if(epochLen_s <= 0):
                Num = 1
            else:
                Num = int(data.shape[1] / (epochLen_s * self.srate))
            for i in range(Num):
                recordTemp = CDataRecord()
#                print(data.shape,i*epochLen_s*self.srate,(i+1)*epochLen_s*self.srate)
                if(epochLen_s <= 0):
                    recordTemp.dataSeg = data
                else:
                    recordTemp.dataSeg = data[:,i*epochLen_s*self.srate:(i+1)*epochLen_s*self.srate]
                if(stimuli.name.lower().find('stimuli1') != -1):
                    recordTemp.label = 0
                elif(stimuli.name.lower().find('stimuli2') != -1):
                    recordTemp.label = 1
                else:
                    print("error, doesn't recognize this kind of stimuli.name", stimuli.name)
                recordTemp.srate = self.srate
                recordTemp.labelDes = labelDes
                if(i*epochLen_s*self.srate<data.shape[1]):
                    dataSetObject.dataRecordList.append(recordTemp)
                
        return dataSetObject
                
    def assignTargetData(self,targetList,frontLag_s = 1, postLag_s = 1): #need to be improved
        ''' only for auditory stimuli evoked data '''
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
                    self.labels[label] = data
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
                        self.labels[label] = data
                        break
                    else:
                        data = data + data1
                        interval[1] = interval1[1]
                
                if(breakFlag == True):
                    break
                
        for to_remove in remove_list:
            self.labels.pop(to_remove,None)
            

class CDataSet:
    def __init__(self,dataSetName = None):
        self.dataRecordList = list()
        self.name = dataSetName
        
    def constructFromFile(self,fileName):
        import pickle
        file = open(fileName, 'rb')
        temp = pickle.load(file)
        
        self.name = temp.name
        self.dataRecordList = temp.dataRecordList
        


        
class CDataRecord: #base class for data with label
    def __init__(self):
        self.dataSeg = None #real data Segment
        self.label = None #segmented "auditoryStimuli" object
        self.labelDes = list() #name of the stimuli ( main)
        self.srate = 0
        self.filterLog = list()
        
    def errorPrint(self,error):
        print("CDataRecorder error" + error)
        
class CAudioDataRecord(CDataRecord):
    
    def constructor(self,data,stimuli):
        self.dataSeg = data
        self.label = stimuli
        self.Check()
    
    def Check(self):
        if(type(self.dataSeg)!=type(np.empty((0,0)))):
            self.errorPrint("type of dataSeg is wrong, should be" + type(np.empty((0,0))))
        if(type(self.label) != type(auditoryStimuli)):
            self.errorPrint("type of dataSeg is wrong, should be" + type(auditoryStimuli))
    
    def errorPrint(self,error):
        print("CAudioDataRecorder error" + error)
        
class ensembleDataSet:
    
    def __init__(self):
        self.dict = dict() # dict for the different DataSet source, like cap-eeg, attentivU-EEG, eog
        self.data = list()
        self.sourceList = list()
        self.labels = list()
    
    def checkDataSet(self):
        
        flag = True
        num_dataSeg_expect = len(self.labels)
        
        for dataOneModal in self.data:
            if(len(dataOneModal) == num_dataSeg_expect):
                return False

        return flag