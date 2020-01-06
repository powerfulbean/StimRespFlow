# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:35:20 2019

@author: Jin Dou
"""
import mne
from mne import concatenate_epochs,create_info
from random import shuffle
import numpy as np

def auditoryLabelParser(labels):
    startEvent = list()
    endEvent = list()
    for label in labels:
        if(label.find('_n_') != -1 and label.find('_Start') != -1):
            startEvent.append(label)
        elif(label.find('_n_') != -1 and label.find('_End') != -1):
            endEvent.append(label)
    
    return startEvent, endEvent

def randEpochSeqNInitTrainData(oEpochs: mne.Epochs, min_trials: int,epochLen_s:int, eventId_dict: dict) -> mne.Epochs:
    event_name = list(eventId_dict.keys())
    print(event_name)
    if (len(event_name)/2 > int(len(event_name)/2)):
        epoch_eventCls1Name_list = event_name[0:int(len(event_name)/2)+1]
        epoch_eventCls2Name_list = event_name[int(len(event_name)/2)+1:len(event_name)]
        print("randEpochSeqNInitTrainData warning")
    else:
        epoch_eventCls1Name_list = event_name[0:int(len(event_name)/2)]
        epoch_eventCls2Name_list = event_name[int(len(event_name)/2):len(event_name)]
    epoch_eventCls1_list = list()
    epoch_eventCls2_list = list()
    
    epoch_eventCls1_list = getEpochBasedOnNames(oEpochs, epoch_eventCls1Name_list)
    epoch_eventCls2_list = getEpochBasedOnNames(oEpochs, epoch_eventCls2Name_list)
    
    epoch_subEventCls1_list = list()
    epoch_subEventCls2_list = list()
    
    epoch_subEventCls1_list = getSubEpochBasedOnSegLen(epoch_eventCls1_list, oEpochs.tmax, epochLen_s)
    epoch_subEventCls2_list = getSubEpochBasedOnSegLen(epoch_eventCls2_list, oEpochs.tmax, epochLen_s)
    shuffle(epoch_subEventCls1_list)
    shuffle(epoch_subEventCls2_list)
    
    trainEpochs_list = epoch_subEventCls1_list[0:min_trials] + epoch_subEventCls2_list[0:min_trials]
    testEpochs_list = epoch_subEventCls1_list[min_trials:] + epoch_subEventCls2_list[min_trials:]
    shuffle(testEpochs_list)
    
#    print(len(trainEpochs_list), len(testEpochs_list))
    finalList = trainEpochs_list + testEpochs_list
    return concatenate_epochs(finalList)
#    return finalList
    
def getEpochBasedOnNames(oEpochs:mne.Epochs, names:list) -> list:
    epoch_eventCls1_list = list()
    for name in names:
        epoch_eventCls1_list.append(oEpochs[name])
    return epoch_eventCls1_list

def getSubEpochBasedOnSegLen(epochsList:list, tmax,epochLen_s:int) -> list:
    print(tmax,epochLen_s)
    tmax = np.ceil(tmax)
    numTrialsPerEpoch = int(tmax / epochLen_s)
    print('numTrialsPerEpoch: ',numTrialsPerEpoch)
    epoch_subEventCls1_list = list()
    for epoch in epochsList:
        for i in range(numTrialsPerEpoch):
            if((i+1) * epochLen_s <= epoch.tmax):
                intv = epoch.times[1] - epoch.times[0]
                epochTemp = epoch.copy().crop(float(i*epochLen_s) , float((i+1) * epochLen_s-intv))
            else:
                epochTemp = epoch.copy().crop(float(i*epochLen_s) , float((i+1) * epochLen_s))
                
            epochTemp.shift_time(0,relative = False)
            print(epoch.get_data().shape,epochTemp.get_data().shape)
            if(len(epochTemp.events) != 0):
                epoch_subEventCls1_list.append(epochTemp)
    
    return epoch_subEventCls1_list

def modifyAnnoNameForSnglSpeech(raw):
    raw.annotations.description[0] = 'cal-by-isaac-Left01-p1_n_none_Start'
    raw.annotations.description[1] = 'cal-by-isaac-Left01-p1_n_none_End'
    raw.annotations.description[2] = 'none_n_cal-by-isaac-Left01-p2_Start'
    raw.annotations.description[3] = 'none_n_cal-by-isaac-Left01-p2_End'
    return raw

def addNecessaryChanls(oRaw):
    stim_data = np.zeros((1, len(oRaw.times)))
    info_stim = create_info(['stim1'], oRaw.info['sfreq'], ['stim'])
    
#    realTime_data = oRaw.times.reshape(1,-1)
#    info_realTime = create_info(['realTime'], oRaw.info['sfreq'], ['stim'])
    
    stim_raw = mne.io.RawArray(stim_data, info_stim)
#    realTime_raw = mne.io.RawArray(realTime_data, info_realTime)
    
    oRaw.add_channels([stim_raw], force_update_info=True)
#    oRaw.add_channels([realTime_raw], force_update_info=True)
    
    return oRaw

def addNecessaryChanlsForEpoch(oEpoch: mne.Epochs):
    
    numEvents = len(oEpoch.events)
    temp = oEpoch.times.reshape(1,1,-1)
    realTime_data = temp
    for i in range(1,numEvents):
        realTime_data = np.concatenate([realTime_data,temp])
#    realTime_data = oEpoch.times.reshape(1,1,-1)
    info_realTime = create_info(['realTime'], oEpoch.info['sfreq'], ['stim'])
    
    realTime_raw = mne.EpochsArray(realTime_data, info_realTime)
    
    oEpoch.add_channels([realTime_raw], force_update_info=True)
    
    return oEpoch

def addNecessaryChanlsForEvoked(oEvoked: mne.Evoked):
    
    temp = oEvoked.times.reshape(1,-1)
    realTime_data = temp
#    realTime_data = oEpoch.times.reshape(1,1,-1)
    info_realTime = create_info(['realTime'], oEvoked.info['sfreq'], ['stim'])
    
    realTime_raw = mne.EvokedArray(realTime_data, info_realTime)
    
    oEvoked.add_channels([realTime_raw], force_update_info=True)
    
    return oEvoked

