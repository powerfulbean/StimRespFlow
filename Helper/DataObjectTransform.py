# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:11:54 2020

@author: Jin Dou
"""
import numpy as np

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

class CDataSetToDataLoader:
    
    def __init__(self):
        from ..DataProcessing.DeepLearning import CPytorch
        from ..DataStruct import DataSet
        self.lib_torch = CPytorch().Lib
        self.lib = DataSet
        
    def __call__(self,*args,**kwargs):
        self.ToDataLoader(*args,**kwargs)
    
    def ToDataLoader(self,Dataset,TorchDataSetType,**Args):
        if(not issubclass(TorchDataSetType,self.lib_torch.utils.data.Dataset)):
            raise ValueError('input TorchDataSetType is not a class of Torch Dataset')
            
        if(not isinstance(Dataset,self.lib.CDataSet)):
            raise ValueError('input Dataset is not a instance of Torch Dataset')
        
        x = Dataset.data.T
        y = Dataset.stimuli.T
        xTensor = self.lib_torch.cuda.FloatTensor(x)
        yTensor = self.lib_torch.cuda.LongTensor(y)
        
        if(Args.get('DataSetArgs') != None):
            DataSetArgs = Args['DataSetArgs']
            dataset = TorchDataSetType(xTensor, yTensor,**DataSetArgs)
        else:
            dataset = TorchDataSetType(xTensor, yTensor)
           
        if(Args.get('DataLoaderArgs') != None):
            DataLoaderArgs = Args['DataLoaderArgs']
            dataLoader = self.lib_torch.utils.data.DataLoader(dataset,**DataLoaderArgs)
        else:
            dataLoader = self.lib_torch.utils.data.DataLoader(dataset)
        
        return dataLoader
        