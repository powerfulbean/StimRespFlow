# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:52:18 2019

@author: Jin Dou
"""

from StellarBrainwav import DataIO
from StellarBrainwav.Helper.Cache import CStimuliCacheAuditory, CStimuliTypeList
from StellarBrainwav.Helper.StageControl import CStageControl
from StellarBrainwav.DataIO import getFileList,saveObject, loadObject, CLog
from StellarBrainwav.DataStruct.Abstract import CData, CRawData,CStimulus
from StellarBrainwav.DataStruct.RawData import CBitalinoRawdata,CDateTimeStampsGen
from StellarBrainwav.DataStruct.LabelData import CVisualLabels,CAuditoryLabels
from StellarBrainwav.DataStruct.DataSet import CDataOrganizor,EOperation,CDataSet
from StellarBrainwav.outsideLibInterfaces import CIfMNE
from StellarBrainwav.DataProcessing import SignalProcessing as SigProc
from StellarBrainwav.DataStruct.Array import CStimuliVectors,CStimulusVector
from StellarBrainwav.DataStruct.StimuliData import CWordStimulus

from StellarBrainwav.BrainwavEngines import StagesEngine
import numpy as np

# oStage = CStageControl([1.1,1.2,1.3,1.4,1.6])
oStage = CStageControl([4.1])
if oStage(1):
    oTemp = CStimuliVectors(3)
    oTemp.append(np.array([11,12,13]))
    oTemp.append(np.array([21,22,23]))
    oTemp.append(np.array([31,32,33]))
    oTemp.append(np.array([41,42,43]))
    
    oTemp[0] = np.ones((oTemp.shape[1],1))
    oTemp[:,1] = np.zeros((1,oTemp.shape[0]))
    print(oTemp)

if oStage(1.1):
    oTemp = CStimuliVectors(3)
    oTemp.append(np.array([11,12,13]))
    oTemp.append(np.array([21,22,23]))
    oTemp.append(np.array([31,32,33]))
    oTemp.append(np.array([41,42,43]))
    print(oTemp.numpy())
    oTemp[0:2] = np.ones((oTemp.shape[1],2))
    print(oTemp.numpy())
    oTemp[:,0:2] = np.zeros((2,oTemp.shape[0]))
    print(oTemp.numpy())
    
if oStage(1.2):
    oTemp = CStimuliVectors(1)
    oTemp.append('I')
    oTemp.append('am')
    oTemp.append('powerfulbean')
    oTemp.append('!')
    oTemp[0] = ['shit'] * 4
    print(oTemp.numpy())
    oTemp[0,0:2] = ['what?'] * 2
    print(oTemp.numpy())
    oTemp[:,0] = np.array([1,2,3])
    print(oTemp.numpy())
    oTemp[:,1] = np.array([1,2,3])
    print(oTemp.numpy())
    oTemp[:,2] = np.array([1,2,3])
    print(oTemp.numpy())
    oTemp[:,3] = np.array([1,2,3])
    print(oTemp.numpy())
    
if oStage(1.3):
    oTemp = CStimuliVectors(1)
    oTemp.append('I')
    oTemp.append('am')
    oTemp.append('powerfulbean')
    oTemp.append('!')
    oTemp[0] = ['shit'] * 4
    print(oTemp.numpy())
    oTemp[0,0:2] = ['what?'] * 2
    print(oTemp.numpy())
    oTemp[:,0:2] = [np.array([1,2,3]),np.array([1,2,3])]
    oTemp[0,2:] = [np.array([2,1,3]),np.array([2,1,3])]
    print(oTemp)
    print(oTemp.numpy())
    oTemp[1,2:] = [np.array([-2]),np.array([-2])]
    print(oTemp)

if oStage(1.4):
    oTemp1 = CStimulusVector([1,2,3])
    
if oStage(1.5):
    oTemp2 = CStimulus([1,2,3])
    
if oStage(1.6):
    otemp3 = CWordStimulus(128)

if oStage(2):
    ''' prepare label and data files'''
    dir_list = ['dirLabels','dirData','dirStimuli','dirResult']
    #oDir = DataIO.DirectoryConfig(dir_list,r"testConf\GlsDataDirectoryVisual.conf")
    oDir = DataIO.CDirectoryConfig(dir_list,r"testConf\GlsDataDirectoryAuditory.conf")
    oDir.checkFolders()
    labelFiles = getFileList(oDir['dirLabels'],'.txt')
    dataFiles = getFileList(oDir['dirData'],'.txt')
    oLog = CLog(oDir['dirResult'],'programLog')
    
    oLog.safeRecordTime('stimuli cache start')#log
    ''' load stimuli in cache '''
    stimuliTypeList = ['MW_DS_NM']
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
    #oRaw.calTimeStamp()
    
    #oLabel = CVisualLabels()
    oLabel = CAuditoryLabels()
    oLabel.readFile(labelFiles[0])
    oLabel.loadStimuli("","cache",oStimCache)
    ''''''
    
    ''' match labels and raw data'''
    oDataOrg = CDataOrganizor(oRaw.numChannels,oRaw.sampleRate,oRaw.description['channelInfo'][1])
    oDataOrg.addLabels(oLabel)
    oDataOrg.assignTargetData([oRaw])
    ''''''

if oStage(3):
    @StagesEngine.stage('test1',1)
    def testFirst0(a):
        print(a)
        
    @StagesEngine.stage('test1',2)
    def testFirst1(a):
        print(a**2,'-1')
        
    @StagesEngine.stage('test2',1)
    def testFirst2(a):
        print(a**2,a-2)
    StagesEngine.paramsLoadFlag = True
    testFirst0(1)
    testFirst1(2)
    testFirst2(3)
    StagesEngine.paramsLoadFlag = False
    StagesEngine.startEngine()
    print()
    with StagesEngine.CStagesEngine(['test2','test1']) as oStagesEngine:
        testFirst0(-1)
        testFirst1(-2)
        testFirst2(-3)
        oStagesEngine.startEngine()
    print()  
    with StagesEngine.CStagesEngine(['test2','test1']) as oStagesEngine:
        testFirst0(-1)
        testFirst1(-2)
        testFirst2(-3)
        oStagesEngine.startEngine(['test1','test2'])
    print()
    testFirst0(-1)
    testFirst1(-2)
    testFirst2(-3)
    oStagesEngine.startEngine(['test1','test2'])
    oStagesEngine.startEngine(['test2','test1'])
    
    
if oStage(4):
    from StellarBrainwav.Helper import StudyManage as sbSM
    tarFolder = r'F:/Dataset/stellarBrainwavTest'
    oStudy = sbSM.CStudy(tarFolder,'test_CStudy',['test1','test2','test3'])
    oStudy.save()
    oExpr = oStudy.newExpr()
    
if oStage(4.1):
    from StellarBrainwav.Helper import StudyManage as sbSM
    paramDict = dict()
    paramDict['test1'] = 1
    paramDict['test2'] = 2
    paramDict['test3'] = 3
    tarFolder = r'F:/Dataset/stellarBrainwavTest'
    oStudy = sbSM.CStudy(tarFolder,'test_CStudy',list(paramDict),('inside_with',max))
    with oStudy.newExpr(paramDict,['test_again_inside_with']) as oLog:
        paramDict['inside_with'] = 15
        paramDict['test_again_inside_with'] = 16
        oLog('test, hello StellarBrainwav')
        # a
        
if oStage(5):
    oDataset = CDataSet()
    oDataset.constructFromFile(r'.\read semantic EEG.bin')
    
    
    
    
#''' Use MNE to preprocess the data'''
#
#nChannels = oDataOrg.n_channels
#channelsList = oDataOrg.channelsList
#sRate = oDataOrg.srate
#
#for label in oDataOrg.labelList:
#    data = oDataOrg[label]
#    
#    #filter and resample the raw data
#    oMNE = CIfMNE(oDataOrg.channelsList,sRate,['eeg','eog'])
#    oMNERaw = oMNE.getMNERaw(data)
#    oMNERaw.filter(2,8,picks = ['eeg'])
#    oMNERaw.filter(0.1,8, picks = ['eog'])
#    oMNERaw.resample(64,npad = 'auto')
#    oDataOrg[label] = oMNERaw.get_data()
#    oDataOrg.srate = oMNERaw.info["sfreq"] #! very important
#    
#    #prepare the envelope of the stimuli
#    mainStream = label.stimuli.mainStream
#    otherStream = label.stimuli.otherStreams[0]
#    mainEnv = SigProc.audioStimPreprocessing(mainStream,8,label.stimuli.len_s,64)
#    otherEnv = SigProc.audioStimPreprocessing(otherStream,8,label.stimuli.otherLen_s[0],64)
#    label.stimuli.mainStream = mainEnv
#    label.stimuli.otherStreams[0] = otherEnv
#
#oDataOrg.logOp('earEEG',EOperation.BandPass,[2,8])
#oDataOrg.logOp('Eog',EOperation.BandPass,[0.1,8])
#oDataOrg.logOp('stimuli',EOperation.LowPass,[8])
#oDataOrg.logOp('stimuli',EOperation.Transform,['hilbert'])
#oDataOrg.logOp('all',EOperation.Resample,[64])
#oDataSet = oDataOrg.getEpochDataSet(60)
##saveObject(oDataOrganizor,r"./",'testOrganizorAuditory')
##temp = loadObject(r"./testOrganizorAuditory.bin")
#''''''
#oLog.safeRecordTime('prepare dataset end')#log




