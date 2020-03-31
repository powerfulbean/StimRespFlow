# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:25:44 2020

@author: Jin Dou
"""
from configparser import ConfigParser
from ..DataIO import readAuditoryStimuli, getFileName
import ast
import numpy as np
from abc import ABC, abstractmethod

class CStimuliCache(ABC):
    
    def __init__(self,dir_Stimuli):
        self.diskRootFolder = dir_Stimuli
        self.cache = dict()
    
    @abstractmethod
    def loadStimuli(self,nameLists,extension):
        pass
    
    @abstractmethod    
    def __getitem__(self,name):
        pass
        
class CStimuliCacheAuditory(CStimuliCache):
    
    
    def loadStimuli(self,nameLists,extension = '.wav'):
        for name in nameLists:
            mainStream, len_s = readAuditoryStimuli(self.diskRootFolder + name + extension) 
            tempSndStreamRecrd = CSoundStreamRecord(mainStream,len_s)
            print(name)
            self.cache[name] = tempSndStreamRecrd
    
    def getStimuli(self,name):
        if (name == 'none'):
            data = [0.00001] * 14400001 
            data = np.array(data)
            return data, 300.0
        else:
            print(len(self.cache[name].soundData),self.cache[name].len_s)
            return self.cache[name].soundData.copy(), self.cache[name].len_s
        
    def __getitem__(self,name):
        if (name == 'none'):
            data = [0.00001] * 14400001 
            data = np.array(data)
            return data, 300.0
        else:
            print(len(self.cache[name].soundData),self.cache[name].len_s)
            return self.cache[name].soundData.copy(), self.cache[name].len_s
        
class CSoundStreamRecord:
    
    def __init__(self,soundData, len_s):
        self.soundData = soundData
        self.len_s = len_s

class CStimuliTypeList:
    
    def __init__(self,stimuliType_List,confFile):
        self.confFile = confFile
        self.type_dict = dict()
        for i in stimuliType_List:
            self.type_dict[i] = ''
        self.load_conf()
        
    def load_conf(self):
        conf_file = self.confFile
        config = ConfigParser()
        config.read(conf_file,encoding = 'utf-8')
        conf_name = getFileName(conf_file)
        for dir_1 in self.type_dict:
            print(dir_1)
            self.type_dict[dir_1] = ast.literal_eval(config.get(conf_name, dir_1))
    
    def getFileList(self,stimuliType):
        return self.type_dict[stimuliType]
    
    def __getitem__(self,stimuliType):
        return self.type_dict[stimuliType]