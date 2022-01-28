# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:57:19 2022

@author: Jin Dou
"""
import os
from abc import abstractmethod,ABC

from .. import outsideLibInterfaces as outLib
from ..DataIO import  getFileName
from .Cache import CStimuliCache
from ..DataStruct.Abstract import CLabels
from ..DataStruct.StimuliData import CAuditoryStimulus
import datetime
import numpy as np


class CVisualLabels(CLabels):
    
    def stimuliEventParser(self,buffer, stimuliStartTag, StimuliEndTag, i, recordObject):
        stimuliStart = buffer[i]
        if(stimuliStart[0] == stimuliStartTag):
            temp = stimuliStart[1]
            realName = temp
            recordObject.name = realName
            
            temp = stimuliStart[2:len(stimuliStart)] # calculate start time
            [h,m,s,ms] = [int(t) for t in temp]
            startTime = datetime.datetime(self.startTime.year,self.startTime.month,self.startTime.day,h,m,s,ms)
            recordObject.startTime = startTime
            i+=1
            
            stimuliEnd = buffer[i]
            #the audio End Label
            if(stimuliEnd[0] == StimuliEndTag):
                recordObject.index = stimuliEnd[1]
                temp = stimuliEnd[2:len(stimuliEnd)] # calculate end time
                [h,m,s,ms] = [int(t) for t in temp]
                endTime = datetime.datetime(self.startTime.year,self.startTime.month,self.startTime.day,h,m,s,ms)
                recordObject.endTime = endTime
                i+=1
            else:
                print("visuallabels, parser error")
        else:
            print("visuallabels, parser error")
        
        return i, recordObject
    
    def readFile(self,fileName, readStimuli = False, stimuliDir = None):
        ''' 
        read all the necessary information from the label files
        
        '''
        #include the OutsideLib module for Input/Output
        self.outLibIO = outLib._OutsideLibIO()
        self.description = 'Visual'
        self.type = "Visual"
        buffer = []
        
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
                    tempRecord = CTimeIntervalStamp('','','','') #store crossing time
                    tempRecord2 = CTimeIntervalStamp('','','','') #store stimuli time
                    tempRecord3 = CTimeIntervalStamp('','','','') #store rest time
                    
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
            
    def loadStimuli(self,Folder, extension, oCache : CStimuliCache = None):
        ''' 
        load stimuli in self.timestamps(CTimeIntervalStamp).stimuli
        '''
        pass
        
class CBlinksCaliLabels(CLabels):
    
    def readFile(self, fileName):
        self.outLibIO = outLib._OutsideLibIO()
        tempName= getFileName(fileName)
        self.description = 'BlinksCali_' + tempName
        self.type = "BlinksCali"
        buffer = []
        with open(fileName, 'r') as the_file: #opoen the labels file
            lines = the_file.readlines()
        
        for line in lines: # read the label file
            temp = line.split()
            buffer.append(temp)
            
        i = 0
        self.startTime = self.parseTimeString(buffer[i][1] + ' ' + buffer[i][2])
        while( i < len(buffer)): #buffer is the whole document
            tempRecord = CTimeIntervalStamp('','','','') #store crossing time
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
    
class CAuditoryLabels(CLabels):       
    
    def readFile(self,fileName):
        ''' 
        read all the necessary information from the label files
        '''
        #include the OutsideLib module for Input/Output
        self.outLibIO = outLib._OutsideLibIO()
        self.description = 'Auditory'
        self.type = "Auditory"
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
                
                tempRecord = CTimeIntervalStamp('','','','')
                tempStimuli = CAuditoryStimulus()
                
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
                        tempStimuli.name = stimuliName
                        #print(stimuliName)
                        otherstimulisName = self.stimuliname(not leftflag, realName)# other stimuli name
                        tempStimuli.otherNames.append(otherstimulisName)
                        #print(otherstimulisName)
                    else:
                        tempStimuli.name = realName
                        tempStimuli.otherNames.append(realName)
                    
                    # calculate start time
                    temp = audioStart[2:len(audioStart)] 
                    [h,m,s,ms] = [int(t) for t in temp]
                    startTime = datetime.datetime(self.startTime.year,self.startTime.month,self.startTime.day,h,m,s,ms)
                    tempRecord.startTime = startTime
                    i+=1
                    
                    #the audio End Label
                    audioEnd = buffer[i]
                    if(audioEnd[0] == 'audioEnd'):
                        tempRecord.index = audioEnd[1]
                        temp = audioEnd[2:len(audioEnd)]
                        [h,m,s,ms] = [int(t) for t in temp]
                        endTime = datetime.datetime(self.startTime.year,self.startTime.month,self.startTime.day,h,m,s,ms)
                        tempRecord.endTime = endTime
                
                tempRecord.type = 'auditory'
                self.append(tempRecord,tempStimuli)
                i+=1
            elif(buffer[i][0]=='-1'):
                break
            else:
                raise TypeError("labels, loadfile error", buffer[i][0])
        
        self.writeType("auditory")              
        return buffer        

    
    def stimuliname(self,isLeft, realName):
        idx = realName.find('R')
        if(isLeft == True):
            realStimuli = realName[1:idx] # don't include the 'L'
        else:
            realStimuli = realName[idx+1:len(realName)] # don't include the 'R'
        return realStimuli    
    
    def loadStimuli(self,Folder, extension, oCache : CStimuliCache = None):
        ''' 
        load stimuli in self.timestamps(CTimeIntervalStamp).stimuli
        '''
        if(extension == '.wav'):
            for i in range(len(self.timestamps)):
                label = self.rawdata[i]
                mainStreamName = label.name
                otherStreamNames = label.otherNames
                print("AuditoryLabels, readStimuli:", mainStreamName,otherStreamNames[0])
                mainStreamFullPath = Folder + mainStreamName + extension
                otherStreamFullPaths = [Folder + i + extension for i in otherStreamNames]
                self.rawdata[i].loadStimulus(mainStreamFullPath,otherStreamFullPaths)
                # save this auditoryStimuli object to the data attribute of this markerRecord object
        
        elif(extension == 'cache'):
            for i in range(len(self.timestamps)):
                label = self.rawdata[i]
                mainStreamName = label.name
                otherStreamNames = label.otherNames
                print("AuditoryLabels, read Stimuli from cache:", mainStreamName,otherStreamNames[0])
                self.rawdata[i].loadStimulus(mainStreamName,otherStreamNames,oCache)                    
                # save this auditoryStimuli object to the data attribute of this markerRecord object
 
