# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:01:24 2018

@author: Samwi
"""
import pywt
import numpy as np
import pyeeg 
import MednovWaveDL as DL

def arrayListCat(List,level,catChannel = None):
    c=[]
    if catChannel == None:
        c = List[0]
        for i in List[1:len(List)]:
            c=np.concatenate([c,i])  
    else:
        catChannel.sort()
        if catChannel[0]<level:
            c = List[catChannel[0]]
        else:
            return
        for i in range(1,len(catChannel)):
            if catChannel[i]>=len(List):
                return
            j=List[catChannel[i]]
            c=np.concatenate([c,j])
    return c

def DWT(data, waveName,catChannel = None,level = None):
    db = pywt.Wavelet(waveName)
    levelMax = pywt.dwt_max_level(len(data[0]),db)
    if level == None:
        level = levelMax
    elif level > levelMax:
        level = levelMax
    else:
        level = level
    mwOutput=[]
    for i in data:
        a=pywt.wavedec(i, db,mode='zpd')
        mwOutput.append(a)
    cat_mWave=[]
    for i in mwOutput:
        temp = arrayListCat(i,level,catChannel)
        temp=temp
        cat_mWave.append(temp)
    return cat_mWave

class ChildNode(object):
    def __init__(self):
        self.dict= [dict(),dict()]
        self.feature=[]
        self.size=[0,0]#用于记录计算的特征的数目

class FeatureNode(object):
    def __init__(self):
        self.trainSet_feat = [[],[]] #trainSet[0] 存波形, trainSet[1] 存图像
        self.trainSet_tag = []
        self.testSet_feat = [[],[]]
        self.testSet_tag = []

class WaveFeature(object):
    def __init__(self,DataSet):
        self.DataSet = DataSet
        self.FeatureSet = ChildNode()
        self.FeatureSet.feature = FeatureNode()
        self.FeatureSet.feature.trainSet_tag = DataSet.trainSet_tag
        self.FeatureSet.feature.testSet_tag = DataSet.testSet_tag
        self.FeatureSet.feature.trainSet_feat[0] = np.empty((len(DataSet.trainSet_data[0]),0))
        self.FeatureSet.feature.testSet_feat[0] = np.empty((len(DataSet.testSet_data[0]),0))
#        self.DLinput_name = 'main_input'
#        self.Auxinput_name = 'aux_input'
#        self.DL_tensor = ''
#        self.DL_input = ''
#        self.Aux_Input = ''
#        self.cat_tensor = ''
#        self.model = ''
    
#    def Run(self,num_classes=2):
#        Output  =   DL.outputTensor(self.cat_tensor,num_classes,0)
#        print(self.cat_tensor.shape)
#        self.model = DL.model(self.DL_input,self.Aux_Input,Output)
#        DL.model_run(self.model,
#                     self.DLinput_name,self.DataSet.trainSet_data[0],
#                     self.Auxinput_name,self.FeatureSet.feature.trainSet_feat[0],
#                     self.FeatureSet.feature.trainSet_tag,
#                     self.DataSet.testSet_data[0],
#                     self.FeatureSet.feature.testSet_feat[0],
#                     self.FeatureSet.feature.testSet_tag)
#
#    def SetTensor(self):
#        self.Aux_Input = DL.eegTFAinput_tensor(self.FeatureSet.size[0],aux_input_name='aux_input')
#        self.DL_input,self.DataSet.trainSet_data[0],self.DataSet.testSet_data[0], self.FeatureSet.feature.trainSet_tag,self.FeatureSet.feature.testSet_tag\
#        = DL.eegDLinput_tensor(self.DataSet.trainSet_data[0],self.FeatureSet.feature.trainSet_tag,self.DataSet.testSet_data[0], self.FeatureSet.feature.testSet_tag)
#        self.DL_tensor = DL.eegLSTM_tensor(self.DL_input,156)
#        self.cat_tensor = DL.catTensor(self.Aux_Input,self.DL_tensor)
#        
    def feature_wave(self, toolName = None, Fs = 256):
        if( toolName == None):
            print('please select a tool')
            return
        
        if toolName in self.FeatureSet.dict[0]:
            index = self.FeatureSet.dict[0][toolName]
        else:
            index = -1
        print(toolName)
        if toolName == 'DWT':
            answer_train = DWT(self.DataSet.trainSet_data[0],'db4')
            answer_test = DWT(self.DataSet.testSet_data[0],'db4')
            print('DWT feature extraction succeed db4')
        elif toolName == 'hurst':
            answer_train = [pyeeg.hurst(i) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.hurst(i) for i in self.DataSet.testSet_data[0]]
            print('hurst feature extraction succeed')
        elif toolName == 'dfa' :
            answer_train = [pyeeg.dfa(i,L=[4,8,16,32,64]) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.dfa(i,L=[4,8,16,32,64]) for i in self.DataSet.testSet_data[0]]
            print('dfa feature extraction succeed')
        elif toolName == 'fisher_info':
            answer_train = [pyeeg.fisher_info(i,2,20) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.fisher_info(i,2,20) for i in self.DataSet.testSet_data[0]]
            print('fisher_info feature extraction succeed')
        elif toolName == 'svd_entropy' :
            answer_train = [pyeeg.svd_entropy(i,2,20) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.svd_entropy(i,2,20) for i in self.DataSet.testSet_data[0]]
            print('svd_entropy feature extraction succeed')
        elif toolName == 'spectral_entropy':
            bandlist=[0.5,4,7,12,30,100]
            answer_train = [pyeeg.spectral_entropy(i,bandlist,Fs) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.spectral_entropy(i,bandlist,Fs) for i in self.DataSet.testSet_data[0]]
            print('spectral_entropy feature extraction succeed')
        elif toolName == 'hjorth':
            # 得到两个量 第一个是 mobility 第二个是 complexity
            answer_train = [pyeeg.hjorth(i) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.hjorth(i) for i in self.DataSet.testSet_data[0]]
            answer_train = np.array(answer_train)
            answer_test = np.array(answer_test)
            
            for i in answer_train:
                i[1] = i[1]/100
            for i in answer_test:
                i[1] = i[1]/100
                
            #只取Mobility
            answer_train = np.array(answer_train[:,0])
            answer_test = np.array(answer_test[:,0])
            print('hjorth feature extraction succeed')
        elif toolName == 'hfd':
            answer_train = [pyeeg.hfd(i,8) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.hfd(i,8) for i in self.DataSet.testSet_data[0]]
            print('hfd feature extraction succeed')
        elif toolName == 'pfd':
            answer_train = [pyeeg.pfd(i) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.pfd(i) for i in self.DataSet.testSet_data[0]]
            print('pfd feature extraction succeed')
        elif toolName == 'bin_power':
            bandlist=[0.5,4,7,12]#,30,100]
            answer_train = [pyeeg.bin_power(i,bandlist,Fs) for i in self.DataSet.trainSet_data[0]]
            answer_test  = [pyeeg.bin_power(i,bandlist,Fs) for i in self.DataSet.testSet_data[0]]
            print('bin_power feature extraction succeed')
            
        else:
            print ('does not have this kind of mode')
        
        answer_train = np.array(answer_train)
        answer_train = answer_train.reshape(len(answer_train),-1)
        answer_test = np.array(answer_test)
        answer_test = answer_test.reshape(len(answer_test),-1)
        if index ==-1:
            #print(len(self.FeatureSet.feature.trainSet_feat[0]),len(answer_train))
            self.FeatureSet.feature.trainSet_feat[0] = np.column_stack((self.FeatureSet.feature.trainSet_feat[0],answer_train))
            self.FeatureSet.feature.testSet_feat[0]=np.column_stack((self.FeatureSet.feature.testSet_feat[0],answer_test))
            self.FeatureSet.dict[0][toolName] = [self.FeatureSet.size[0],self.FeatureSet.size[0]+len(answer_train[0])]
            self.FeatureSet.size[0]+= len(answer_train[0])
        else:
            self.FeatureSet.feature.trainSet_feat[0][:,index[0]:index[1]] = [ i for i in answer_train]
            self.FeatureSet.feature.testSet_feat[0][:,index[0]:index[1]] = [i for i in answer_test]
        
        
    def feature_img(self,toolName = None):
        if( toolName == None):
            print('please select a tool')
            return
       
    def get_Feature(self,featName,Type):
        if Type == 'wave':
            Offset = 0
        elif Type == 'img':
            Offset = 1
        else:
            print('MednovFeatEx warning: dont have this type')
            
        Index = self.FeatureSet.dict[Offset][featName]
        return self.FeatureSet.feature.trainSet_feat[Offset][:,Index[0]:Index[1]],self.FeatureSet.feature.testSet_feat[Offset][:,Index[0]:Index[1]]
    
    def get_MulFeature(self,WavefeatList = None , ImgfeatList = None):
        Mul_feat_train = np.empty((len(self.DataSet.trainSet_data[0]),0))
        Mul_feat_test = np.empty((len(self.DataSet.testSet_data[0]),0))
        
        if WavefeatList.all() != None:
            for i in WavefeatList:
                train_temp,test_temp = self.get_Feature(i,'wave')
                Mul_feat_train = np.column_stack((Mul_feat_train,train_temp))
                Mul_feat_test = np.column_stack((Mul_feat_test,test_temp))
        
        if ImgfeatList.all() != None:
            for i in ImgfeatList:
                img_train_temp,img_test_temp = self.get_Feature(i,'wave')
                img_Mul_feat_train = np.column_stack((Mul_feat_train,train_temp))
                img_Mul_feat_test = np.column_stack((Mul_feat_test,test_temp))
            
        if WavefeatList.any() == None and ImgfeatList.any() != None:
            return img_Mul_feat_train,img_Mul_feat_test
        elif WavefeatList.any() != None and ImgfeatList.any() == None:
            return Mul_feat_train,Mul_feat_test
        else:
            return Mul_feat_train,Mul_feat_test,img_Mul_feat_train,img_Mul_feat_test
    
    def get_Tag(self):
        return self.FeatureSet.feature.trainSet_tag,self.FeatureSet.feature.testSet_tag