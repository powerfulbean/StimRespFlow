# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:41:55 2021

@author: Jin Dou
"""

import numpy as np
from scipy.stats import zscore
import mne

def lalorlabDetectEEGBadChannels(eegarray,verbose = True):
    '''
    we assume the first dimension is channel dimension

    Parameters
    ----------
    eegarray : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    eegarray = np.array(eegarray)
    assert len(eegarray.shape) == 2 
    stdChans = list()
    badChansIdx = list()
    for chan in eegarray:
        stdChans.append(np.std(chan))
    
    for idx,chan in enumerate(eegarray):
        if np.std(chan) > 2.5 * np.mean(stdChans):
            badChansIdx.append(idx)
        
    stdChans.clear()
    
    for idx,chan in enumerate(eegarray):
        if idx not in badChansIdx:
            stdChans.append(np.std(chan))
    
    for idx,chan in enumerate(eegarray):
        if np.std(chan) < np.mean(stdChans) / 2.5:
            badChansIdx.append(idx)
        
    if verbose:
        print(badChansIdx)
    
    return badChansIdx

def mneWrapLalorlabDetectEEGBadChannels(mneraw:mne.io.RawArray, montage = None, nNearest = 10):
    oRaw = mneraw.copy()
    data = oRaw.get_data()
    if montage is None:
        badChansIdx = lalorlabDetectEEGBadChannels(data,False)
    else:
        badChansIdx = lalorlabDetectBadChannelsByCovVarNear(data,montage, nNearest = nNearest)
    oRaw.info['bads'] = [oRaw.info['ch_names'][i] for i in badChansIdx]
    print(f'bad channels: {",".join(oRaw.info["bads"])}')
    return oRaw


def lalorlabDetectBadChannelsByCovVarNear(data, montage, th1 = 2, th2 = 2, nNearest = 6):
    # data: (nChan, nSamples)
    data = np.array(data)
    assert data.ndim == 2
    
    ### prepare the nearest channels
    if nNearest > 0:
        chanloc = montage.get_positions()['ch_pos']
        chnames = []
        poses = []
        for n,pos in chanloc.items():
            chnames.append(n)
            poses.append(pos)
            
        assert data.shape[0] == len(chnames)
        chanDistMat = np.zeros((len(chanloc), len(chanloc)))
        
        fDist = lambda pos1,pos2: np.sqrt(np.sum((pos1 - pos2)**2))
        
        for i in range(len(chnames)):
            for j in range(len(chnames)):
                chanDistMat[i,j] = fDist(poses[i], poses[j])
        
        nearChanIdx = []
        for i in range(len(chnames)):
            nearChanIdx.append(np.argsort(chanDistMat[i])[1:nNearest+1])
    else:
        nearChanIdx = [None] * data.shape[1]
    ### find the bad channels
    dataz = zscore(data, axis = 1)
    XTX = np.matmul(dataz ,dataz.T)
    stdXTX = np.std(XTX, axis = 1)
    stdEEG = np.std(data, axis = 1)
    
    badChans = []
    if nNearest <=0 :
        badChans.append(np.where(stdXTX < np.mean(stdXTX) / th1))
        badChans.append(np.where(stdEEG > np.mean(stdEEG) * th2))
    else:
        for chanIdx in range(data.shape[0]):
            # print(stdXTX[chanIdx],
            #       stdEEG[chanIdx],
            #       np.mean(stdXTX[nearChanIdx[chanIdx]]) / th1,
            #       np.mean(stdEEG[nearChanIdx[chanIdx]]) * th2)
            if stdXTX[chanIdx] < np.mean(stdXTX[nearChanIdx[chanIdx]]) / th1:
                badChans.append(chanIdx)
            if stdEEG[chanIdx] > np.mean(stdEEG[nearChanIdx[chanIdx]]) * th2:
                badChans.append(chanIdx)
            
    return list(set(badChans))


    
def plotChanWithNamesAtIdx(montage, idxs):
    chnames = montage.ch_names
    montage.plot(show_names = [chnames[idx] for idx in idxs])
    
            
# def      
        
    
        