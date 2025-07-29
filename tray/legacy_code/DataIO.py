# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:45:32 2019

@author: Sam Jin Dou
"""

from configparser import ConfigParser,BasicInterpolation
import struct 
import array,os
import numpy as np
from .outsideLibInterfaces import _OutsideLibIO
from .outsideLibInterfaces import _OutsideLibTime

'''[start] load different kinds of data files '''
def loadBin(Dir):
    f = open(Dir,'rb')
    data=[]
    
    while True:
        b = f.read(8)
        if(len(b)==0):
            break
        else:
            data.append(struct.unpack('d',b)) 
            
    caldata=[float(p[0]) for p in data]
    f.close()
    return caldata

def loadBinFast(Dir):
    f = open(Dir,'rb')
    a = array.array('d')
    a.fromfile(f, os.path.getsize(Dir) // a.itemsize)
    return np.asarray(a)

def loadJpg(Dir,i = None, iMax = None):
    from PIL import Image
    return np.array(Image.open(Dir))[:,:,2]

def loadText(Dir):
    f=open(Dir, "r")
    contents = f.read()
    f.close()
    return contents

def loadSound(Dir,Fs=22050):
    import librosa  
    wav,sr = librosa.load(Dir,Fs)
    return wav

def readAuditoryStimuli(file):
    ''' 
    read audio stimuli
    '''
    print("readStimuli:", file)
    frameNum, channelNum, data, len_s = _OutsideLibIO().getMonoChannelData(file)
    a = _unpackWav(frameNum,channelNum,data)
    
    return np.asarray(a), len_s

def _unpackWav(frameNum, channelNum, data):
    return struct.unpack("%ih" % (frameNum* channelNum), data)

''' End '''

''' File and Folder Related Operation'''
def getFileList(folder_path,extension):
    out = [folder_path+file for file in os.listdir(folder_path) if file.endswith(extension)]
    if(len(out)==0):
        print("getFileList's error: folder: '" + str(folder_path) + "'is empty with '" + str(extension) + "' kind of file")
        return -1
    else:
        return out

def labelFileRecog(file_path):
    Type = ['auditory','visual','attention','blinkCali']
    with open(file_path, 'r') as the_file: #opoen the labels file
        lines = the_file.readlines()
            
    line = lines[1].split()
    if (line[0] == 'left' or line[0] == 'right' or line[0] == 'Single'):
        return Type[0]
    elif (line[0] == 'First' or line[0] == 'Cat'):
        return Type[1]
    elif (line[0] == 'P'):
        return Type[2]
    elif (line[0] == 'blink' or line[0] == 'lookLeft' or line[0] == 'lookRight'):
        return Type[3]
    else:
        return -1
    
def checkFolder(folderPath):
    if not os.path.isdir(folderPath):
        print("path: " + folderPath + "doesn't exist, and it is created")
        os.makedirs(folderPath)
        
def getFileName(path):
    dataSetName, extension = os.path.splitext(os.path.basename(path))    
    return dataSetName    

def getSubFolderName(folder):
    subfolders = [f.name for f in os.scandir(folder) if f.is_dir() ]
    return subfolders

''' End '''

''' Python Object IO'''
def saveDataSetObject(Object,folderName):
    checkFolder(folderName)
    print(Object.name)
    file = open(folderName + Object.name.replace(':','_') + '.bin', 'wb')
    import pickle
    pickle.dump(Object,file)
    file.close()
    
def saveObject(Object,folderName,tag, ext = '.bin'):
    checkFolder(folderName)
    file = open(folderName + str(tag) + ext, 'wb')
    import pickle
    pickle.dump(Object,file)
    file.close()
    
def loadObject(filePath):
    import pickle
    file = open(filePath, 'rb')
    temp = pickle.load(file)
    return temp

''' End '''

''' Log IO'''
class CLog:
    
    def __init__(self,folder,Name):
        checkFolder(folder)
        self.fileName = folder+Name + '.txt'
        self.Open()
        self.save()
        self.folder = folder
        self.name = Name
        self._logable = True
        
    def setLogable(self,logFlag):
        self._logable = logFlag
        
    def getLogable(self):
        return self._logable 
    
    def Open(self):
        self.fileHandle = open(self.fileName, "a+")
        
    def record(self,log,newline:bool = True):
        if(self.getLogable() == False):
            return
        
        if(type(log)==list):
            for j in log:
                self.fileHandle.write(str(j)+' ')
        elif(type(log) == str):
            self.fileHandle.write(log)
        else:
            print("Clog doesn't support this kind of log", type(log))
        
        if(newline == True):
            self.fileHandle.write('\n')
        else:
            self.fileHandle.write('\t')
        
    def openRecordSave(self,log,newline:bool = True):
        if(self.getLogable() == False):
            return
        self.Open()
        self.record(log,newline)
        self.save()
    
    def safeRecord(self,log,newline:bool = True):
        if(self.getLogable() == False):
            return
        self.openRecordSave(log ,newline)
		
    def safeRecordTime(self,log,newline:bool = True):
        if(self.getLogable() == False):
            return
        datetime = _OutsideLibTime()._importDatetime()
        self.openRecordSave(log + ', time: ' + str(datetime.datetime.now()),newline)
    
    def save(self):
        self.fileHandle.close()

''' End '''

''' Data Folder Directory Configuration Related'''
class CDirectoryConfig:
    
    def __init__(self,dir_List, confFile):
        self.dir_dict = dict()
        self.confFile = confFile
        for i in dir_List:
            self.dir_dict[i] = ''
        self.load_conf()
            
    def load_conf(self):
        conf_file = self.confFile
        config = ConfigParser(interpolation=BasicInterpolation())
        config.read(conf_file,encoding = 'utf-8')
        conf_name = getFileName(conf_file)
        for dir_1 in self.dir_dict:
            self.dir_dict[dir_1] = config.get(conf_name, dir_1)
#            print(dir_1,config.get(conf_name, dir_1))
    
    def p(self,keyName):
        return self.dir_dict[keyName]
    
    def __getitem__(self,keyName): 
        ''' overloading function for [] operator '''
        return self.dir_dict[keyName]
    
    def checkFolders(self,foldersList = None):
        if foldersList != None:
            pass
        else:
            foldersList = self.dir_dict.keys()
        
        for folder in foldersList:
                checkFolder(self.p(folder))
            
''' End '''
            
            
