# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 01:26:24 2019

@author: Jin Dou
"""
import outsideLibInterfaces as outLib
from RawData import CRawData
from abc import ABC, abstractmethod, ABCMeta
from DataIO import checkFolder, getFileName

class CLabels(CRawData): # for labels created by psychopy, the last number is microsecond but not millisecond              
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
    
    @abstractmethod
    def readFile(self,fileName):
        ''' not useful for labels, because label's data can be loaded later'''
        return

        
class BlinksCaliLabels(CLabels):
    
    def readFile(self, fileName):
        self.outLibIO = outLib._OutsideLibIO()
        tempName= getFileName(fileName)
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
            tempRecord = CLabelInfo('','','','') #store crossing time
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

class CLabelInfo:
    ''' class to store information for a label marker, including:
        1. label name
        2. other useful label names
        3. start and end time
        4. label type
        5. label value
    '''
    def __init__(self,name,index,startTime,endTime):
        self.name = name
        self.otherNames = list() # background stimuli name
        self.index = index
        self.startTime = startTime #datetime.time or datetime.datetime object
        self.endTime = endTime #dateime.time or datetime.datetime object 
        self.type = ''
        self._promoteFlag = False
        self._promoteTimeFlag = False
        self.value = '' #label info (such as auditoryStimuli object)
        
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
            self.value = data
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
    
if __name__ == '__main__':
    test2 = BlinksCaliLabels()
    test2.writeInfoForMatlab(r'.')