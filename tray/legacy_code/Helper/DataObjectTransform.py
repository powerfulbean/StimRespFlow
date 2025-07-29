# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:11:54 2020

@author: Jin Dou
"""
import numpy as np
from abc import ABC,abstractmethod
import mne
from packaging import version
# from typing import Tuple

class CTransformBase(ABC):
    
    def __call__(self,*args,**kwargs):
        return self.trans(*args,**kwargs)
    
    @abstractmethod
    def trans(self):
        pass
        
class CEEGLabToMNE(CTransformBase):
    
    @staticmethod
    def chanlocs(eeglabChanlocs:dict): #-> tuple[list[str],mne.channels.DigMontage]:
        from mne.io.eeglab import eeglab
        locs:list[dict] = eeglab._dol_to_lod(eeglabChanlocs)
        class temp:
            pass
        oTemp = temp()
        oTemp.chanlocs = locs
        
        if version.parse(mne.__version__) < version.parse("0.20.0"):
            chNames,montage = eeglab._get_eeg_montage_information(oTemp, True)
            return chNames, montage
        else:
            chNames,ch_types,montage = eeglab._get_montage_information(oTemp, True)
            return chNames,montage,ch_types
    
    @staticmethod
    def readFormalFile(path):
        montage = mne.channels.read_custom_montage(path)
        return montage.ch_names, montage

class CEpochToDataLoader:
    
    def __init__(self):
        from ..DataProcessing.DeepLearning import CPytorch
        from ..outsideLibInterfaces import CIfMNE
        self.lib_MNE = CIfMNE().LibMNE
        self.lib_torch = CPytorch().Lib
    
    def __call__(self,epoch,picks:list=None,shuffle=True,expandAxis = 1):
        return self.ToCUDADataLoader(epoch,picks,shuffle,expandAxis)
        
    
    def ToCUDADataLoader(self,epoch,picks:list=None,shuffle=True,expandAxis = 1):
        from sklearn.preprocessing import  LabelEncoder,minmax_scale
        if(not isinstance(epoch,self.lib_MNE.Epochs)):
            raise ValueError('input epoch is not a instance of MNE Epochs')
        
        x = epoch.get_data(picks)
#        List = list()
#        for i in range(x.shape[1]):
#            temp = x[:,i,:]
#            temp = minmax_scale(temp,feature_range=(0, 1),axis = 1)
#            temp = np.expand_dims(temp,1)
#            List.append(temp)
#        x = np.concatenate(List,axis = 1)
        le = LabelEncoder()
        y = le.fit_transform(epoch.events[:, 2])
        x = np.expand_dims(x,expandAxis)
        xTensor = self.lib_torch.cuda.FloatTensor(x)
        yTensor = self.lib_torch.cuda.LongTensor(y)
        dataset = self.lib_torch.utils.data.TensorDataset(xTensor, yTensor)
        dataLoader = self.lib_torch.utils.data.DataLoader(dataset,shuffle=shuffle,batch_size=100)
        return dataLoader


class TensorsToDataLoader(ABC):
    
    def __init__(self):
        from ..DataProcessing.DeepLearning import CPytorch
        self.lib_torch = CPytorch().Lib
        
    def __call__(self,*args,**kwargs):
        pass
    
class CToTensors(ABC):
    
    def __init__(self):
        import torch
        self.lib_torch = torch
    
    def __call__(self,*args,**kwargs):
        #return tuple
        return self.toTensors(*args,**kwargs)
    
    @abstractmethod
    def toTensors(self):
        pass
    
class CDataRecordToTensors(CToTensors):
    
    def __init__(self):
        super().__init__()
        from ..DataStruct import DataSet
        self.lib = DataSet
    
    def toTensors(self,DataRecord,T = True):
        if(not isinstance(DataRecord,self.lib.CDataRecord)):
            raise ValueError('input Dataset is not a instance of CDataRecord')
        if T:
            x = DataRecord.data.T
            y = DataRecord.stimuli.T
        else:
            x = DataRecord.data
            y = DataRecord.stimuli
        xTensor = self.lib_torch.FloatTensor(x)
        yTensor = self.lib_torch.FloatTensor(y)
        
        return xTensor,yTensor
    
class CNArraysToTensors(CToTensors):
    
    def __init__(self):
        super().__init__()
    
    def toTensors(self,*arrays,T = True):
        if T:
            output = [self.lib_torch.FloatTensor(i.T) for i in arrays]
        else:
            output = [self.lib_torch.FloatTensor(i) for i in arrays]
        return output

    
    
class CRawDataToTensors(CToTensors):
    
    def __init__(self):
        super().__init__()
        from ..DataStruct import RawData
        self.lib = RawData
    
    def toTensors(self,RawData):
        assert isinstance(RawData, self.lib.CRawData)
        
        x = RawData.rawdata.T
        xTensor = self.lib_torch.FloatTensor(x)
        
        return (xTensor,)

        