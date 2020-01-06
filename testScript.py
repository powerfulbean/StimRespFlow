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
from DataStruct.DataSet import CDataOrganizor

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
oDataOrganizor = CDataOrganizor(oRaw.numChannels,oRaw.sampleRate,oRaw.description['channelInfo'][1])
oDataOrganizor.addLabels(oLabel)
oDataOrganizor.assignTargetData([oRaw])

saveObject(oDataOrganizor,r"./",'testOrganizorAuditory')

#temp = loadObject(r"./testOrganizor.bin")
