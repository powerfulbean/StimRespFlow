# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:23:08 2018

@author: Samwi
"""
import os
import struct 
from PIL import Image
import numpy as np
import sys,time
import librosa


class _DataNode(object):
    def __init__(self):
        self.data =[''] 
        self.index=[]
        self.trainSet_data = [''] 
        self.trainSet_tag = ['']
        self.testSet_data = ['']
        self.testSet_tag = ['']
    
class _ListNode(object):
    def _init_self(self):
        self.filelist = []
        self.index = []
        self.dict = dict()#用于存储文件名称和列表索引的映射的字典

class DataSet(object):
    def __init__(self):
        self.DataList = _ListNode()
        self.DataSet = _DataNode()
        self.flag = 0
        
    def buildFilelist (self, Object, root_filedir_list,suffix):
        filelist=[]
#        index=[]
        if(len(root_filedir_list)==1):
            file_dir = root_filedir_list[0]
            filelist_per=[]
            for root, dirs, files in os.walk(file_dir): 
                for i in files:
                    Suffix_up = suffix.upper()
                    Suffix_low = suffix.lower()
                    if i.endswith(Suffix_low) or i.endswith(Suffix_up):
                        temp = os.path.join(root,i)
                        filelist_per.append(temp) #当前路径下所有非目录子文件
            filelist = filelist_per
        elif(len(root_filedir_list)>0):  
            for file_dir in root_filedir_list:
                filelist_per=[]
                for root, dirs, files in os.walk(file_dir): 
                    for i in files:
                        Suffix_up = suffix.upper()
                        Suffix_low = suffix.lower()
                        if i.endswith(Suffix_low) or i.endswith(Suffix_up):
                            temp = os.path.join(root,i)
                            filelist_per.append(temp) #当前路径下所有非目录子文件
                filelist.append(filelist_per)
        Object.filelist = filelist
        print("filelist createion success")
    
    def buildDictPerCate (self, Object):#针对某一类的数据建立字典dict()
        Dict = dict()
        for i in range(len(Object.filelist)):
            Dict[self.getFileName(Object.filelist[i])]=i
        Object.dict = Dict
        print("Dictionary createion success")
    
    def buildDictWaveImg(self, Object):
        Dict = dict()#第一列存波形的索引，第二列存图像的索引
        for i in Object.wave.dict.keys():
            Dict[Object.wave.dict[i]]=Object.img.dict[i[len(i)-6]+'_'+self.cutSuffix(i)+'.jpg']  #未来需要改一下，目前文件名命名格式不标准
        Object.dict_WaveImg = Dict
        
    def getFileName(self,Str):
        Len = len(Str)
        for i in range(Len):
            if Str[Len-i-1]=='\\':
                return Str[Len-i:Len]
    
    def cutSuffix(self,Str):
        Len = len(Str)
        for i in range(Len):
            if Str[Len-i-1]=='.':
                return Str[0:Len-i-1]               
    
    def buildDataSet(self,labellist,mode=None,len_trainset=3000,len_testset=25,Size = 256):#只载入指定大小的且经过打乱的文件的数据
#        需要补充多分类驱动器
#        index = self.DataList.index
        filelist = self.DataList.filelist
        
        trainList_x= []
        testList_x = []
        trainList_y=np.empty(0)
        testList_y=np.empty(0)
        
        if len(labellist)==len(filelist):
            for i in range(len(filelist)):
                trainList_x_temp, testList_x_temp = self._buildRandomList(filelist[i],len_trainset,len_testset)
                trainList_y_temp, testList_y_temp = self._getLabel(trainList_x_temp,testList_x_temp,[labellist[i]])
                trainList_x+=trainList_x_temp
                testList_x+=testList_x_temp
                trainList_y=np.concatenate((trainList_y,trainList_y_temp),0)
                testList_y=np.concatenate((testList_y,testList_y_temp),0)
            
               
#        trainList_y,testList_y = self._getLabel(trainList_x,testList_x,'samename','.trn')
        
        x_train = []
        y_train = []
        x_test = []
        y_test=[]
        
        x_train,x_test = self._loadDataSet(trainList_x,testList_x,'bin')
        y_train = np.array(trainList_y)
        y_test = np.array(testList_y)
            
        
        self.DataSet.trainSet_data = x_train
        self.DataSet.trainSet_tag = y_train
        self.DataSet.testSet_data = x_test
        self.DataSet.testSet_tag = y_test
            
        self.DataSet.trainSet_data,self.DataSet.trainSet_tag=self.reIndex(self.DataSet.trainSet_data,self.DataSet.trainSet_tag)
            
        print('build DataSet Succeed')
    
    def reIndex(self,x_train,y_train):
        #打乱数据集
        if len(x_train) == 0:
            return '',''
        index = [i for i in range(len(x_train))]    
        np.random.shuffle(index)   
        x_train = x_train[index]  
        y_train = y_train[index]
        
        return x_train,y_train
    
    def _getLabel(self,filelist1,filelist2,label,label_suffix=None):
        
        
        if(len(label)) == 1:
            y_train = np.zeros(len(filelist1), dtype=int)
            y_test = np.zeros(len(filelist2), dtype=int)
            for i in range(len(y_train)):
                y_train[i] = label[0]
            for i in y_test:
                y_test[i] = label[0]
        elif label == 'samename':
            y_train = [i+label_suffix for i in filelist1]
            y_test = [i+label_suffix for i in filelist2]
        
        return y_train,y_test
        
    def _buildRandomList(self,filelist,len_trainset,len_testset):
        
        len_data = len(filelist)
            
        randlist=np.random.randint(1,len_data,len_trainset+len_testset)
        list_data=[filelist[i] for i in randlist]
            
        trainList1=[]
        testList1=[]
                
        trainList1=list_data[0:len_trainset]
        testList1 =list_data[len_trainset:len_trainset+len_testset]
        
        return trainList1, testList1
      
    def _loadDataSet(self,trainList1,testList1,mode,Size = 256):
        
        x_train=[]
        x_test=[]

#        count = 0
#        countMax = len(trainList1)+len(testList1)
#        for i in trainList1:#先读取txt文件，再根据其对应的名称读入相应的图片文件
#            caldata1=self.loadTxt(i,Size)
#            x_train.append(caldata1)   
#            sys.stdout.write("\rLoading data for training progress: %d%%   " % int(100*count/countMax))
#            sys.stdout.flush()
#            count+=1
#        
#        for i in trainList1:#先读取txt文件，再根据其对应的名称读入相应的图片文件
#            sys.stdout.write("\rLoading data for training progress: %d%%   " % int(100*count/countMax))
#            sys.stdout.flush()
#            count+=1
        
        if mode=="bin":#/"Txt"/"TXT":
            x_train = [self.loadBin(wave,Size) for wave in trainList1]
            x_test  = [self.loadBin(wave,Size) for wave in testList1]
        
        elif mode == "jpg":
            x_train = [self.loadJpg(img) for img in trainList1]
            x_test  = [self.loadJpg(img) for img in testList1]
        
        elif mode == 'sound':
            x_train = [self.loadSound(sound) for sound in trainList1]#序列长度不一样
            x_test  = [self.loadSound(sound) for sound in testList1]
        elif mode == 'utf8':
            x_train = [librosa.load(sound) for sound in trainList1]
            x_test  = [librosa.load(sound) for sound in testList1]
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        
        return x_train, x_test
        
        
    def loadSound(self,Dir,Fs=22050):
        wav,sr = librosa.load(Dir,Fs)
        return wav
    
    def loadBin(self,Dir,Num = 256):
        f = open(Dir,'rb')
        data=[]
        for k in range(Num):
            b=f.read(8)
            data.append(struct.unpack('d',b))       
        caldata=[float(p[0]) for p in data]
        f.close()
        return caldata
    
    def loadJpg(self,Dir,i = None, iMax = None):
        return np.array(Image.open(Dir))[:,:,2]
    
    def testDict(self):
        for i in self.DataList.dict_WaveImg.keys():
            j=self.getFileName(self.DataList.wave.filelist[i])
            j1 = self.getFileName(self.DataList.img.filelist[self.DataList.dict_WaveImg[i]])
            if (j[len(j)-6]+'_'+self.cutSuffix(j)) !=  self.cutSuffix(j1):
                print(1)
    
class thchsDataSet(DataSet):
    def __init__(self):
        self.DataList = _ListNode()
        self.DataSet = _DataNode()
        self.flag = 0
