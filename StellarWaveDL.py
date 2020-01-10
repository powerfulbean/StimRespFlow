# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:20:29 2018

@author: Jin Dou
"""
import keras
from keras.layers import Dense, Flatten,Input,LSTM,Activation
from keras.layers import Conv1D, Conv2D,MaxPooling1D,Dropout,normalization,MaxPooling2D
import matplotlib.pylab as plt
from keras.models import Model
from keras.layers.core import  Reshape


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc=[]
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        


class DeepLearning (object):
    def __init__(self,batch_size = 1000,epochs = 100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.CNN_layer = [] #存储CNN每个层的对象
        self.LSTM_layer = [] #存储LSTM每个层的对象
    
    def patternGene(a,b=None,c=None,d=None):
        pattern = list()
        if a!=None :
            pattern.append(a)
        if b!=None :
            pattern.append(b)
        if c!=None :
            pattern.append(c)
        if d!=None :
            pattern.append(d)
        return pattern

    def setSuPara(self,batch_size = 1000,epochs = 100):
        self.batch_size = batch_size
        self.epochs = epochs
    
    def buildInput(self,input_shape,Name = None):
        tensor_1 = Input(shape=input_shape,name = Name)
        return tensor_1
        
    '''
    buildCNN
    作者：JinDou wechat:powefulbean
    函数功能:针对keras框架，用于解析一个模式列表，并根据这个列表搭建对应的神经网络的函数
    输入:tensor_input:作为输入的张量
        :patternList:一个网络配置的模式列表
    返回值：输出的张量  
    !!!!
    函数的核心是在内部让不同层首尾相连
    !!!
    
    “单个层配置列表”:解析“单个模式列表”需要调用getTensor函数，规定“单个模式列表”的第一个元素代表层的类型
    例:
    ['Cov2D',16,(3,3),(1,1),'relu'] 就是一个 “单个层配置列表”
    
    把多个 “层配置列表” 封装在一起，构成一个 “结构模式” 
    “结构模式”: [“单个层配置列表1”,...,“单个层配置列表N”] 目前N最大为4
    
    模式列表：[
          [“结构模式1”,“结构模式重复次数1"],
          ...
          [“结构模式N”,“结构模式重复次数N"],
      ]
    '''
    def buildCNN(self,tensor_input,patternList):# patternList :    [ [ pattern1 ,num1 ],...,[ patternN ,numN ]]
        if len(patternList )== 0:
            print ('CNN error:empty patternlist')
            return
        
        tensor_1 = tensor_input
        tensor_Out = ''
        for index,i in enumerate(patternList): # i = [ [['name1',....,]['name2',....,]['name3',....,]],num ]
            pattern = i[0] #获取一个[结构模式,num]中的“结构模式”
            num = i[1]#获取[结构模式,num]中 “结构模式”重复的次数
            if index == 0:
                numh_tensor = tensor_1#若是整个列表中的第一个[结构模式,num]，则把tensor_input赋值给这个[结构模式,num]的头张量
            else:
                numh_tensor = tensor_Out#若不是整个列表中的第一个[结构模式,num]，则把上一个[结构模式,num]的输出张量赋值给当前[结构模式,num]的头张量
            numl_tensor = ''#创建当前[结构模式,num]的尾张量
            for j in range(num):#开始根据“结构模式”重复的次数num进行循环
                if j == 0:
                    path_tensor = numh_tensor#若是当前整个[结构模式,num]中的第一个 “结构模式”，则给当前“结构模式”的头张量赋值当前整个[结构模式,num]的头张量
                else:
                    path_tensor = numl_tensor#若不是整个[结构模式,num]中的第一个 “结构模式”，则给当前第j次下重复这个“结构模式”的头张量赋值第j-1次重复这个结构模式的尾张量
                patl_tensor = ''
                for k in pattern: # pattern=[['name1',....,]['name2',....,]['name3',....,]],K为“单个配置列表”
                    temp= self.getTensor(k)#解析 “单个配置列表”
                    self.CNN_layer.append(temp)
                    patl_tensor = temp(path_tensor)#获取当前“单个配置列表”的尾张量
                    path_tensor = patl_tensor#给当前“结构模式”内的下一个“单个配置列表”的头张量赋值当前“单个配置列表”的尾张量
                numl_tensor = patl_tensor#用当前[结构模式,num]的尾张量保存当前第j次重复这个“结构模式”得到的尾张量
            tensor_Out = numl_tensor#用整个函数的输出张量保存当前[结构模式,num]得到的尾张量
        
        return tensor_Out#返回最后一个结构模式的输出张量作为整个配置列表的输出张量
    
    '''
    buildNN解析一个模式列表，和buildCNN完全相同
    '''
    def buildNN(self,tensor_input,patternList):
        # patternList :    [ [ pattern1,num1 ],...,[ patternN ,numN ]]
        if len(patternList )== 0:
            print ('CNN error:empty patternlist')
            return
        
        tensor_1 = tensor_input
        tensor_Out = ''
        for index,i in enumerate(patternList): # i = [ [['name',....,][][]],num ]
            pattern = i[0]
            num = i[1]
            if index == 0:
                numh_tensor = tensor_1
            else:
                numh_tensor = tensor_Out
            numl_tensor = ''
            for j in range(num):
                if j == 0:
                    path_tensor = numh_tensor
                else:
                    path_tensor = numl_tensor
                patl_tensor = ''
                for k in pattern: # j=[['name',....,][][]]
                    #print(k)
                    temp= self.getTensor(k)
                    self.CNN_layer.append(temp)
                    patl_tensor = temp(path_tensor)
                    path_tensor = patl_tensor
                numl_tensor = patl_tensor
            tensor_Out = numl_tensor
        
        return tensor_Out
    
    def buildLSTM(self,tensor_input,state_num = 156):
        temp = LSTM(state_num)
        self.LSTM_layer.append(temp)
        tensor_2 = temp(tensor_input)
        return tensor_2
        
        
    def getTensor(self,p):
        if p[0] == 'Cov2D':
            return Conv2D(p[1], kernel_size=p[2], strides=p[3],activation=p[4])
        elif p[0] == 'Cov1D':
            return Conv1D(p[1], kernel_size=p[2], strides=p[3],activation=p[4])
        elif p[0] == 'MaxPool':
            return MaxPooling2D(pool_size=p[1], strides=p[2])
        elif p[0] == 'MaxPool1D':
            return MaxPooling1D(pool_size=p[1], strides=p[2])
        elif p[0] == 'Dropout':
            return Dropout(p[1])
        elif p[0] == 'lstm':
            return LSTM(p[1])
        elif p[0] == 'dense':
            return Dense(p[1],activation = p[2])
        elif p[0] == 'flatten':
            return Flatten()
        elif p[0] == 'reshape':
            return Reshape((p[1]))
        elif p[0] == 'activ':
            return Activation(p[1])
        else:
            print('dont support this layer')
    
    def catTensor(self,tensor1,tensor2):#连接两个张量
        return keras.layers.concatenate([tensor1,tensor2])
    
    def get_classFormTagset(self,num_classes,y_train=None,y_test=None):#获取符合分类问题格式要求的数据集标签
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        return y_train,y_test
        
    def get_lstmFormDataset(self,x_train=None,x_test=None,seg_len=256):#获取符合lstm格式要求的数据集数据
        if x_train.any() == None or x_test.any() == None  :
            print('MednovWaveDL error: the input is none')         
        else:
            x_train = x_train.reshape(x_train.shape[0],-1,seg_len)
            x_test = x_test.reshape(x_test.shape[0],-1,seg_len)
            
        return x_train,x_test
     
        
    def get_cnnFormDataset(self,x_train=None,x_test=None):#获取符合2Dcnn格式要求的数据集数据
        if x_train.any() == '' or x_test.any() == ''or x_train.any() == None or x_test.any() == None:
            print('MednovWaveDL error: the input is none') 
            return x_train,x_test
        if x_train.any() == None or x_test.any() == None :
            print('MednovWaveDL error: the input is none')         
        else:
            x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
            x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
            
        return x_train,x_test 
    
    def get_1DcnnFormDataset(self,x_train=None,x_test=None):#获取符合1Dcnn格式要求的数据集数据
        if x_train.any() == None or x_test.any() == None  :
            print('MednovWaveDL error: the input is none') 
            return x_train,x_test
        if x_train.any() == None or x_test.any() == None :
            print('MednovWaveDL error: the input is none')         
        else:
            x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
            x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
            
        return x_train,x_test 
                    
            
        
    

