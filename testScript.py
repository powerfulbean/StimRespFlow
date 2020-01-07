# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:52:18 2019

@author: Jin Dou
"""

import DataIO
from Helper.Cache import CStimuliCacheAuditory, CStimuliTypeList
from DataIO import getFileList,saveObject, loadObject
from DataStruct.RawData import CBitalinoRawdata
from DataStruct.LabelData import CVisualLabels,CAuditoryLabels
from DataStruct.DataSet import CDataOrganizor,EOperation
from outsideLibInterfaces import CMNE

''' prepare label and data files'''
dir_list = ['dirLabels','dirData','dirStimuli']
#oDir = DataIO.DirectoryConfig(dir_list,r"testConf\GlsDataDirectoryVisual.conf")
oDir = DataIO.DirectoryConfig(dir_list,r"testConf\GlsDataDirectoryAuditory.conf")
oDir.checkFolders()
labelFiles = getFileList(oDir['dirLabels'],'.txt')
dataFiles = getFileList(oDir['dirData'],'.txt')

''' load stimuli in cache '''
stimuliTypeList = ['MW_DS_NM','MM_SS_NM','MM_DS_WM','MW_DS_WM','WW_SS_NM']
oStimList = CStimuliTypeList(stimuliTypeList,r'testConf/stimuliType.conf')
oStimCache = CStimuliCacheAuditory(oDir['dirStimuli'])
oStimCache.loadStimuli(oStimList['MW_DS_NM'])

'''load label and raw data files'''
oRaw = CBitalinoRawdata()
oRaw.readFile(dataFiles[0],mode = 'EEGandEOG')
oRaw.calTimeStamp()

#oLabel = CVisualLabels()
oLabel = CAuditoryLabels()
oLabel.readFile(labelFiles[0])
oLabel.loadStimuli("","cache",oStimCache)
oLabel.enhanceTimeStamps()

''' match labels and raw data'''
oDataOrg = CDataOrganizor(oRaw.numChannels,oRaw.sampleRate,oRaw.description['channelInfo'][1])
oDataOrg.addLabels(oLabel)
oDataOrg.assignTargetData([oRaw])

''' Use MNE to preprocess the data'''

nChannels = oDataOrg.n_channels
channelsList = oDataOrg.channelsList
sRate = oDataOrg.srate

for label in oDataOrg.labelList:
    data = oDataOrg[label]
    oMNE = CMNE(data,oDataOrg.channelsList,sRate,['eeg','eog'])
    oMNERaw = oMNE.getMNERaw()
    oMNERaw.filter(2,8,picks = ['eeg'])
    oMNERaw.filter(0.1,8, picks = ['eog'])
    oMNERaw.resample(64,npad = 'auto')
    oDataOrg.logOp(label,'earEEG',EOperation.BandPass,[2,8])
    oDataOrg.logOp(label,'Eog',EOperation.BandPass,[0.1,8])
    oDataOrg.logOp(label,'all',EOperation.Resample,[64])
    oDataOrg[label] = oMNERaw.get_data()
    
    
    



#saveObject(oDataOrganizor,r"./",'testOrganizorAuditory')

#temp = loadObject(r"./testOrganizorAuditory.bin")
