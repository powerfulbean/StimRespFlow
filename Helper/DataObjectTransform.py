# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:11:54 2020

@author: Jin Dou
"""
import numpy as np
from abc import ABC,abstractmethod

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
        from ..DataProcessing.DeepLearning import CPytorch
        self.lib_torch = CPytorch().Lib
    
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
    
    def toTensors(self,DataRecord):
        if(not isinstance(DataRecord,self.lib.CDataRecord)):
            raise ValueError('input Dataset is not a instance of Torch Dataset')
        
        x = DataRecord.data.T
        y = DataRecord.stimuli.T
        xTensor = self.lib_torch.FloatTensor(x)
        yTensor = self.lib_torch.FloatTensor(y)
        
        return xTensor,yTensor
    
    
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

        