# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:27:33 2020

@author: Jin Dou
"""
import numpy as np

class CPytorch:
    
    def __init__(self):
        self.Lib = self._ImportTorch()
    
    def _ImportTorch(self):
        import torch as root
        return root
    
    def _getNNAttr(self,name:str):
        import torch.nn as NN
        ans = getattr(NN,name)
        return ans
    
    def fitClassificationModel(self,model,dataLoader,testDataLoader,
             numEpochs:int,lr:float,weight_decay:float,oLossFunc = None):
        criterion = None
        if(oLossFunc == None):
            criterion = self.Lib.nn.CrossEntropyLoss()
        else:
            criterion = oLossFunc
        optimizier = self.Lib.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.cuda()
        step_list = list()
        loss_list = list()
        metrics = list()
        for epoch in range(numEpochs):
            accuList = list()
            model.train()
            loss = None
            for idx,data in enumerate(dataLoader):
                eeg,trainLabel = data
#                eeg.cuda()
                trainLabel.cuda()
#                eeg = self.Lib.autograd.Variable(eeg.cuda())
                # forward
                output = model(eeg)
                loss = criterion(output, trainLabel)
                # backward
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
                train_ep_pred = model(eeg)
                train_accuracy = self.get_accuracy(train_ep_pred, trainLabel)
                accuList.append(train_accuracy)
                if(idx % 10 == 0):
                    print("data: {}, train loss is {}, train accu is {} \n".format((idx), loss.data,np.mean(accuList)))
            self.Lib.cuda.empty_cache()   
            for param_group in optimizier.param_groups:
                print(param_group['lr'])
            test_loss_list = list()
            accuListTest = list()
            for data in testDataLoader:
                eeg1,testLabel = data
                eeg1.cuda()
                testLabel.cuda()
                # forward
                output1 = model(eeg1)
                loss1 = criterion(output1, testLabel)
                test_loss_list.append(loss1.cpu().data.numpy())
                test_accuracy = self.get_accuracy(output1, testLabel)
                accuListTest.append(test_accuracy)
            
            print("epoch: {}, loss is {}, test loss is {}, test accu is {}\n".format((epoch+1), loss.data,np.mean(test_loss_list),np.mean(accuListTest)))
    #        print(test_loss_list)
            if epoch in [numEpochs * 0.125, numEpochs * 0.5, numEpochs * 0.75]:
                for param_group in optimizier.param_groups:
                    param_group['lr'] *= 0.1
                # ·
#            model.eval()
#            train_data_x,train_data_y = dataLoader.dataset.tensors
#            test_data_x,test_data_y = testDataLoader.dataset.tensors
#            train_ep_pred = model(train_data_x)
#            test_ep_pred = model(test_data_x)
#            
#            train_accuracy = self.get_accuracy(train_ep_pred, train_data_y)
#            test_accuracy = self.get_accuracy(test_ep_pred, test_data_y)
#            print("train acc: {}\t test acc: {}\t at epoch: {}\n".format(train_accuracy,
#                                                                     test_accuracy,
#                                                                     epoch))
#            step_list.append(epoch)
#            loss_list.append(loss.cpu().data.numpy())
#            
#            metrics.append([train_accuracy,test_accuracy])
            
        return metrics
    
    def trainClassificationModel(self,model,dataLoader,testDataLoader,numEpochs:int,
                                 lr:float,weight_decay:float,oLossFunc = None):
        criterion = None
        if(oLossFunc == None):
            criterion = self.Lib.nn.CrossEntropyLoss().cuda()
        else:
            criterion = oLossFunc
        optimizier = self.Lib.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model = model.cuda()
        step_list = list()
        loss_list = list()
        metrics = list()
        for epoch in range(numEpochs):
            accuList = list()
            model.train()
            loss = None
            for idx,data in enumerate(dataLoader):
                eeg,trainLabel = data
#                print(eeg,trainLabel)
#                shapeList = list()
#                for i in range(2,len(eeg.shape)):
#                    shapeList.append(i)
#                eeg = eeg.permute(1,0,*shapeList)
#                eeg.cuda()
#                trainLabel.cuda()
#                eeg = self.Lib.autograd.Variable(eeg.cuda())
                # forward
                output = model(eeg)
#                print(output.shape,trainLabel.shape)
                loss = criterion(output, trainLabel)
                # backward
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
#                train_ep_pred = model(eeg)
                
                if(idx % 100 == 0):
                    train_accuracy = self.get_onehot_accuracy(output, trainLabel)
                    accuList.append(train_accuracy)
                    print("data: {}, train loss is {}, train accu is {} \n".format((idx), loss.data,train_accuracy))
            
            self.Lib.cuda.empty_cache()   
            for param_group in optimizier.param_groups:
                print(param_group['lr'])
    #        print(test_loss_list)
            test_loss_list = list()
            accuListTest = list()
            for data in testDataLoader:
                    eeg1,testLabel = data
#                    eeg1.cuda()
#                    testLabel.cuda()
                    # forward
                    output1 = model(eeg1)
                    loss1 = criterion(output1, testLabel)
                    test_loss_list.append(loss1.cpu().data.numpy())
                    test_accuracy = self.get_onehot_accuracy(output1, testLabel)
                    accuListTest.append(test_accuracy)
                
            print("epoch: {}, loss is {}, test loss is {}, test accu is {}\n".format((epoch+1), loss1.data,np.mean(test_loss_list),np.mean(accuListTest)))
            
            if epoch in [numEpochs * 0.125, numEpochs * 0.5, numEpochs * 0.75]:
                for param_group in optimizier.param_groups:
                    param_group['lr'] *= 0.1
            
            
            
            metrics.append([loss.cpu().detach().numpy(),loss1.cpu().detach().numpy(),np.mean(accuList),np.mean(accuListTest)])
            self.Lib.cuda.empty_cache()
        return metrics
    
    def get_accuracy(self,output, targets):
        """calculates accuracy from model output and targets
        """
        output = output.detach()
        predicted = output.argmax(-1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / output.size(0) * 100
        
        return accuracy
    
    def get_onehot_accuracy(self,output,targets):
        output = output.ge(0.5).float()
        correct = (output == targets).float().sum()
        accuracy = correct / (len(output) * output.shape[1])
#        print(output,targets)
        accuracy = accuracy.cpu().numpy()
        return accuracy
    
class CTorchNNYaml(CPytorch):
    
    def __init__(self):
        super().__init__()
        
    def _readYaml(self,filePath):
        import yaml
        ans = None
        with open(filePath,'r') as stream:
            try:
                ans = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return ans
        
    def _ParseType(self,conf:dict):
        if(conf['Type'] == 'Sequential'):
            return self.buildSequential(conf)
        
    def _subListToTuple(self,oInput):
        if type(oInput) == dict:
            for key in oInput:
                if(type(oInput[key]) == list):
                    oInput[key] = tuple(oInput[key])
        
        elif type(oInput) == list:
            for idx,attr in enumerate(oInput):
                if type(attr) == list:
                    oInput[idx] = tuple(attr)
        
        else:
            raise ValueError("_subListToTuple: input should be dict or list")
                
    
    def buildSequential(self,conf:dict):
        oSeq = self.Lib.nn.Sequential()
        ModelConfList = conf['Model']
        for idx,ModelConf in enumerate(ModelConfList):
            CModule = self._getNNAttr(ModelConf[0])
            attr = ModelConf[1]
            oModule = None
            name = str(idx)
                
            if(len(ModelConf) > 2 and type(ModelConf[2]) == dict):
                '''if contain aux attribute'''
                auxAttr = ModelConf[2]
                if (auxAttr.get('name')!=None):
                    ''' if aux attribute contain name attribute'''
                    name = auxAttr['name']
            if(type(attr) == list):
                if len(attr) == 0:
                    oModule = CModule()
                elif(type(attr[0]) == list and type(attr[1]) == dict):
                    self._subListToTuple(attr[0])
                    self._subListToTuple(attr[1])
                    oModule = CModule(*attr[0],**attr[1])
                elif(any(type(x) not in [int,float,str,bool,list] for x in attr)):
                    raise ValueError('attribute of Module %s (index %d) is invalid' % (ModelConf[0],idx))
                else:
                    self._subListToTuple(attr)
                    oModule = CModule(*attr)
            elif(type(attr) == dict):
                self._subListToTuple(attr)
                oModule = CModule(**attr)
            else:
                raise ValueError('attribute of Module %s (index %d) is invalid' % (ModelConf[0],idx))
            oSeq.add_module(name,oModule)
        return oSeq
    
    def __call__(self,confFile:str):
        yamlDict = self._readYaml(confFile)
        return self._ParseType(yamlDict)
    
        
        
        