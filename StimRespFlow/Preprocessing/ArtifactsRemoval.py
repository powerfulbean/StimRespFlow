# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:41:55 2021

@author: ShiningStone
"""

import numpy as np
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

def mneWrapLalorlabDetectEEGBadChannels(mneraw:mne.io.RawArray):
    oRaw = mneraw.copy()
    data = oRaw.get_data()
    badChansIdx = lalorlabDetectEEGBadChannels(data,False)
    oRaw.info['bads'] = [oRaw.info['ch_names'][i] for i in badChansIdx]
    print(f'bad channels: {",".join(oRaw.info["bads"])}')
    return oRaw
    
            
            
        
    
        