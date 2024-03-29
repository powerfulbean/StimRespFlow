# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:52:18 2019

@author: Jin Dou
"""

from StimRespFlow import DataIO
from StimRespFlow.Helper.Cache import CStimuliCacheAuditory, CStimuliTypeList
from StimRespFlow.Helper.StageControl import CStageControl
from StimRespFlow.DataIO import getFileList,saveObject, loadObject, CLog
from StimRespFlow.DataStruct.Abstract import CData, CRawData,CStimulus
from StimRespFlow.DataStruct.WaveData import CBitalinoWaveData,CDateTimeStampsGen
from StimRespFlow.DataStruct.LabelData import CVisualLabels,CAuditoryLabels
from StimRespFlow.DataStruct.DataSet import CDataOrganizor,EOperation,CDataSet
from StimRespFlow.outsideLibInterfaces import CIfMNE
from StimRespFlow.DataProcessing import SignalProcessing as SigProc
from StimRespFlow.DataStruct.Array import CStimuliVectors,CStimulusVector
from StimRespFlow.DataStruct.StimuliData import CWordStimulus

from StimRespFlow.BrainwavEngines import StagesEngine
import numpy as np

# oStage = CStageControl([1.1,1.2,1.3,1.4,1.6])
oStage = CStageControl([7])
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
    
if oStage(2.1):
    from StimRespFlow.DataStruct.WaveData import CBitalinoWaveData
    '''load label and raw data files'''
    dir_list = ['dirLabels','dirData','dirStimuli','dirResult']
    oDir = DataIO.CDirectoryConfig(dir_list,r"testConf\GlsDataDirectoryAuditory.conf")
    dataFiles = getFileList(oDir['dirData'],'.txt')
    oRaw = CBitalinoWaveData()
    oRaw.readFile(dataFiles[0],mode = 'EEGandEOG')

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
    from StimRespFlow.Helper import StudyManage as sbSM
    tarFolder = r'F:/Dataset/StimRespFlowTest'
    oStudy = sbSM.CStudy(tarFolder,'test_CStudy',['test1','test2','test3'])
    oStudy.save()
    oExpr = oStudy.newExpr()
    
if oStage(4.1):
    from StimRespFlow.Helper import StudyManage as sbSM
    paramDict = dict()
    paramDict['test1'] = 1
    paramDict['test2'] = 2
    paramDict['test3'] = 3
    tarFolder = r'F:/Dataset/StimRespFlowTest'
    oStudy = sbSM.CStudy(tarFolder,'test_CStudy',list(paramDict),('inside_with',max))
    with oStudy.newExpr(paramDict,['test_again_inside_with']) as oLog:
        paramDict['inside_with'] = 15
        paramDict['test_again_inside_with'] = 16
        oLog('test, hello StimRespFlow')
        # a
if oStage(4.2):
    from StimRespFlow.Helper import StudyManage as sbSM
    studyRoot = r'D:\OneDrive\ShiningStoneResearch\Language\FinetuneNLPCorrelatesEEG\result'
    oStudy = sbSM.CStudy(studyRoot, '19subj_onset_0ms_700ms', [],('bestCorr',max))
    print(oStudy.getNewExprIndex())
        
        
if oStage(5):
    oDataset = CDataSet()
    oDataset.constructFromFile(r'.\read semantic EEG.bin')
    
    
if oStage(6):
    from StimRespFlow.DataStruct.Array import CWaveArray
    import numpy as np
    array = np.ndarray((2,100))
    oData = array.view(CWaveArray)
    oWave = CWaveArray(10)
    oWave.resize((1,2))
    oWave.sort()
    # oWave.append(np.ndarray((10,2)))
    print(oWave)
    oWave2 = CWaveArray(10,np.ndarray((10,2)))
    print(oWave2)
    oWave2 = CWaveArray(2,[[1,2,3],[1,2,3]])
    print(oWave2)
    oWave2 = CWaveArray(10,np.ndarray((11,2)))
    print(oWave2)
    
if oStage(7):
    # test op trace engine
    from StimRespFlow.BrainwavEngines import OpTraceEngine as sbOE
    class Test(sbOE.MixinTraceable):
        @sbOE.MixinTraceable.tracedOp('print_add',['b'])
        def add(self,a,b):
            print('add = ',a+b)
    
    class Test2():
        @sbOE.MixinTraceable.tracedOp('print_add')
        def add(self,a,b):
            print('add = ',a+b)
            
            
    oTemp = Test()
    oTemp2 = Test2()
    # oTemp.add.doc += '1'
    oTemp.add(1,b=2)
    oTemp2.add(1,2)
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




