# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:17:05 2020

@author: Jin Dou
"""
import numpy as np

class CProtocol:
    
    def __init__(self):
        pass
    
class CDataSetProtocol(CProtocol):
    
    def __init__(self):
        super(CDataSetProtocol,self).__init__()
        
    def check_DataOrganizorLite(self, oOrg):
        '''
        CDataOrganizorLite's DataRecord's data should be simple numpy array data
            its first dimension should indicate the channel's index or class's index of the stimuli.
        CDataOrganizorLite's DataRecord's stimuli should be simple numpy array data,
            its first dimension should indicate the channel's index.
            
        there should be no time overlaps between its different DataRecords
        '''
        name = "CDataOrganizorLite"
        keys = oOrg.getSortedKeys()
        for idx,key in enumerate(keys):
            oRecord = oOrg.dataDict[key]
            if(not isinstance(oRecord.data,np.ndarray) \
                or not isinstance(oRecord.stimuli, np.ndarray)):
                raise ValueError("should be numpy ndarray")
            
            if(len(oRecord.data.shape)<2 or len(oRecord.stimuli.shape)<2
               or oOrg.nChannels != len(oRecord.data)):
                raise ValueError("the dimentions of " + name + "'s data.data and " +
                                 "data.stimuli should be bigger than one")
            if(idx < len(keys) - 1):
                if(oOrg.keyFunc != None): 
                    thisCmpKey = oOrg.keyFunc(key[1])
                    nextCmpKey = oOrg.keyFunc(keys[idx+1][0])
                    if(thisCmpKey >= nextCmpKey):
                        return ValueError("the timestamps should not be overlapped")
                else:
                    thisCmpKey = key[1]
                    nextCmpKey = keys[idx+1][0]
                    if(thisCmpKey >= nextCmpKey):
                        return ValueError("the timestamps should not be overlapped")
        
        return True