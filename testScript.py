# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:52:18 2019

@author: Jin Dou
"""

import DataIO
from Helper.Cache import CStimuliCacheAuditory, CStimuliTypeList
from DataIO import getFileList,saveObject, loadObject, CLog
from DataStruct.RawData import CBitalinoRawdata
from DataStruct.LabelData import CVisualLabels,CAuditoryLabels
from DataStruct.DataSet import CDataOrganizor,EOperation
from outsideLibInterfaces import CMNE
import SignalProcessing as SigProc


''' prepare label and data files'''
dir_list = ['dirLabels','dirData','dirStimuli','dirResult']
#oDir = DataIO.DirectoryConfig(dir_list,r"testConf\GlsDataDirectoryVisual.conf")
oDir = DataIO.DirectoryConfig(dir_list,r"testConf\GlsDataDirectoryAuditory.conf")
oDir.checkFolders()
labelFiles = getFileList(oDir['dirLabels'],'.txt')
dataFiles = getFileList(oDir['dirData'],'.txt')
oLog = CLog(oDir['dirResult'],'programLog')

oLog.safeRecordTime('stimuli cache start')#log
''' load stimuli in cache '''
stimuliTypeList = ['MW_DS_NM','MM_SS_NM','MM_DS_WM','MW_DS_WM','WW_SS_NM']
oStimList = CStimuliTypeList(stimuliTypeList,r'testConf/stimuliType.conf')
oStimCache = CStimuliCacheAuditory(oDir['dirStimuli'])
oStimCache.loadStimuli(oStimList['MW_DS_NM'])
''''''
oLog.safeRecordTime('stimuli cache end')#log

oLog.setLogable(False)
oLog.safeRecord('useless')
oLog.setLogable(True)

oLog.safeRecordTime('prepare dataset start')#log
'''load label and raw data files'''
oRaw = CBitalinoRawdata()
oRaw.readFile(dataFiles[0],mode = 'EEGandEOG')
oRaw.calTimeStamp()

#oLabel = CVisualLabels()
oLabel = CAuditoryLabels()
oLabel.readFile(labelFiles[0])
oLabel.loadStimuli("","cache",oStimCache)
oLabel.enhanceTimeStamps()
''''''

''' match labels and raw data'''
oDataOrg = CDataOrganizor(oRaw.numChannels,oRaw.sampleRate,oRaw.description['channelInfo'][1])
oDataOrg.addLabels(oLabel)
oDataOrg.assignTargetData([oRaw])
''''''

''' Use MNE to preprocess the data'''

nChannels = oDataOrg.n_channels
channelsList = oDataOrg.channelsList
sRate = oDataOrg.srate

for label in oDataOrg.labelList:
    data = oDataOrg[label]
    
    #filter and resample the raw data
    oMNE = CMNE(data,oDataOrg.channelsList,sRate,['eeg','eog'])
    oMNERaw = oMNE.getMNERaw()
    oMNERaw.filter(2,8,picks = ['eeg'])
    oMNERaw.filter(0.1,8, picks = ['eog'])
    oMNERaw.resample(64,npad = 'auto')
    oDataOrg[label] = oMNERaw.get_data()
    oDataOrg.srate = oMNERaw.info["sfreq"] #! very important
    
    #prepare the envelope of the stimuli
    mainStream = label.stimuli.mainStream
    otherStream = label.stimuli.otherStreams[0]
    mainEnv = SigProc.audioStimPreprocessing(mainStream,8,label.stimuli.len_s,64)
    otherEnv = SigProc.audioStimPreprocessing(otherStream,8,label.stimuli.otherLen_s[0],64)
    label.stimuli.mainStream = mainEnv
    label.stimuli.otherStreams[0] = otherEnv

oDataOrg.logOp('earEEG',EOperation.BandPass,[2,8])
oDataOrg.logOp('Eog',EOperation.BandPass,[0.1,8])
oDataOrg.logOp('stimuli',EOperation.LowPass,[8])
oDataOrg.logOp('stimuli',EOperation.Transform,['hilbert'])
oDataOrg.logOp('all',EOperation.Resample,[64])
oDataSet = oDataOrg.getEpochDataSet(60)
#saveObject(oDataOrganizor,r"./",'testOrganizorAuditory')
#temp = loadObject(r"./testOrganizorAuditory.bin")
''''''
oLog.safeRecordTime('prepare dataset end')#log
