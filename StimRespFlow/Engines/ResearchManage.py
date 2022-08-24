# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:53:24 2022

@author: ShiningStone
"""
import os
import datetime
from StellarInfra import DirManage as siDM
from StellarInfra import IO as siIO
from StellarInfra.Logger import CLog

STUDY_FILE_NAME = 'STUDY.txt'
EXPR_FILE_NAME = 'EXPR.txt'
RUN_FILE_NAME = 'RUN.txt'
RUN_FILE_PRIMARY_KEY = 'run_index'

class CResearch:
    pass
    
    
class CStudy:
    #define the scheme of the study variable for different experiments
    
    def __init__(self,root,name,configScheme:list):
        '''

        Parameters
        ----------
        root : str
            DESCRIPTION.
        configScheme : [[],...[]], nested list or single list
            hiearchical configs
            sublist or single list will be used to create one level of folder for expr
        name : str
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        assert isinstance(configScheme,list)
        self.root = root
        self.name = name
        self.configScheme = configScheme
        self.folderLevels = []
        if isinstance(configScheme[0],str):
            self.folderLevels.append(configScheme)
        else:
            for i in configScheme:
                self.folderLevels.append(i)
        self.oNameConfig = siDM.CNameByConfig(includeNone=False)
    
    def genExprFolder(self,configs):
        exprFolder = self.root + '/' + self.name
        configsUsedByHost = {}
        for folderKeys in self.folderLevels:
            curConfig = {k:configs[k] for k in folderKeys}
            exprFolder = exprFolder + '/' + self.oNameConfig(curConfig)
            configsUsedByHost.update(curConfig)
        configs = {k:v for k,v in configs.items() if k not in configsUsedByHost}
        return exprFolder,configsUsedByHost,configs
    
class CExpr:
    #record the results of the study variable of a specific experiments
    def __init__(self,host,configs:dict = None,fMetric:callable = None):
        #host can be either a folder or a CStudy instance
        #fMetric: method to work on the configs that 
        configsUsedByHost = None
        if isinstance(host, CStudy):
            hostFolder,configsUsedByHost,configs = host.genExprFolder(configs)
        elif isinstance(host, str):
            hostFolder = host
        else:
            raise NotImplementedError
        
        exprFolder = siDM.CPath(hostFolder)
        self.fMetric = fMetric
        self.configs = configs
        self.name = os.path.basename(exprFolder)
        self.folder = exprFolder
        print(exprFolder / EXPR_FILE_NAME)
        if not siDM.checkExists(exprFolder / EXPR_FILE_NAME):
            assert configs is not None
            doc = dict.fromkeys(configs.keys())
            doc['config'] = dict()
            doc['config']['expr_name'] = self.name
            if configsUsedByHost is not None:
                doc['config'].update(configsUsedByHost)
            doc['run_list'] = []
            doc['result'] = {}
            self._doc = doc
        else:
            self._doc = siIO.loadJson(exprFolder / EXPR_FILE_NAME)
    
    @property
    def doc(self):
        return self._doc
    
    @property
    def bestExprIdx(self):
        # self.summary()
        idx = self.doc['best_run_indices'][-1]
        return (self.name, idx)
    
    def getNewExprIndex(self):
        return len(self._doc['run_list'])
    
    def summary(self):
        if self.fMetric is not None:
            bestValue,idx = self.fMetric(self._doc['run_list'])
            self.doc["best_run_indices"] = idx
            self.doc['config']["best_run_metrics"] = self.fMetric.__name__
    
    def save(self):
        self.summary()
        siIO.saveDictJson(self.folder / EXPR_FILE_NAME, self._doc)
       
    def __setitem__(self,key,value):
        self.doc[key] = value
        
    def newRun(self,updatedConfigs):
        for k,v in updatedConfigs.items():
            self.configs[k] = v
        
        return CRun(self, self.getNewExprIndex(),self.configs)
    
    
class CRun:
    #record the results of one run, the average of several runs provide result 
    #of an experiment
    
    def __init__(self,parent:CExpr,key,configs):
        self.parent = parent
        self.key = key
        self.starttime = None
        self.endtime = None
        self.configs = configs
        self._oLog = CLog(self.parent.folder / str(key),'/' + RUN_FILE_NAME,'')
        self.folder = self._oLog.folder
        
    def start(self):
        # self._oLog.Mode = 'fast'
        self._oLog.ifPrint = False
        self.starttime = datetime.datetime.now()
        self._oLog("start time:",self.starttime)
        
    def end(self):
        self._oLog.Mode = 'safe'
        self.endtime = datetime.datetime.now()
        self._oLog("end time:",self.endtime)
        
    def append(self):
        #save to experiment log excel file
        #save to .expr file
        expr_record = {}
        expr_record[RUN_FILE_PRIMARY_KEY] = self.key
        expr_record['startTime'] = str(self.starttime)
        expr_record['endTime'] = str(self.endtime)
        expr_record.update(self.configs)
        self.parent.doc['run_list'].append(expr_record)
        self.parent.save()
        self._oLog('CRun Append Finish')
        
    def __setitem__(self,k,v):
        self.configs[k] = v
    
    @property
    def oLog(self):
        return self._oLog
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self,*args):
        self.end()
        #append result only when exits successfully
        if all([i is None for i in args]):
            self.append()
        else:
            self.oLog.t(args[1])    
    