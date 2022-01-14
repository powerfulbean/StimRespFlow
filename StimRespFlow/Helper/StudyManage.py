# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:15:15 2021

@author: ShiningStone
"""
import datetime
from abc import ABC,abstractmethod

from StellarInfra import DirManage as siDM
from StellarInfra import IO as siIO
from StellarInfra.Logger import CExprLogger,CLog

STUDY_FILE_NAME = '.study'
EXPR_FILE_NAME = '.expr'
EXPR_LOG_FILE_NAME = '_expr.xlsx'
EXPR_LOG_FILE_PRIMARY_KEY = 'expr_index'
EXPR_SUM_FILE_PRIMARY_KEY = 'study_tag'

StudyKeys = ['config','experiment_list']
ExperimentKeys = [EXPR_LOG_FILE_PRIMARY_KEY,'startTime','endTime']

class CStudyPathBase(ABC):
    '''
    Used for preparaing necessary data and path
    '''
    
    def __init__(self,confFile,*args,FolderName = None,**kwargs):
        self.oPath = siDM.CPathConfig(confFile,*args,checkFolders = False,**kwargs)
        if FolderName:
            self.config(FolderName)
        else:
            self.config()
        
    @abstractmethod
    def config(self,**kwargs):
        pass

    @abstractmethod
    def study(self,key):
        return {'root':None,'tag':None}
    
    def getBestExpr(self,keyForStudy):
        paths = self.study(keyForStudy)
        bestIdx =  _CStudy_EasyConfig(paths).bestExprIdx[1]
        return paths['root'] / paths['tag'] / bestIdx
        
    

class CExprFile(CExprLogger):
    def __init__(self,file:str):
        self._load([], file)
        

class CStudySummaryExprLogger(CExprLogger):
    def __init__(self,file:str):
        newKeysList = [EXPR_SUM_FILE_PRIMARY_KEY]
        self._load(newKeysList, file)
        self._df = self.df[0:0]#clear all existing rows
        
    def append(self,data):
        data = data.copy()
        self._df = self._df.append(data,ignore_index = True)
        self.save()

class CStudyExprLogger(CExprLogger):
    def __init__(self,oStudy,keys:list,file:str):
        assert isinstance(oStudy, CStudy)
        newKeysList = [EXPR_LOG_FILE_PRIMARY_KEY] + keys
        self._load(newKeysList, file)
        
    def append(self,data:dict,expr_key):
        data = data.copy()
        data[EXPR_LOG_FILE_PRIMARY_KEY] = expr_key
        self._df = self._df.append(data,ignore_index = True)
        self.save()


class CExpr:
    def __init__(self,oStudy,expr_key,dataToAppend:dict = None,keysIncludedInStudyFile = None):
        assert isinstance(oStudy, CStudy)
        self.oParentStudy = oStudy
        self._oLog = CLog(self.oParentStudy.studyPath / str(expr_key),'/',EXPR_FILE_NAME)
        self.expr_key = expr_key
        self.starttime = None
        self.endtime = None
        self.dataToAppend:dict = dataToAppend
        self.keysIncludedInStudyFile = keysIncludedInStudyFile
        
    def start(self):
        # self._oLog.Mode = 'fast'
        self.starttime = datetime.datetime.now()
        self._oLog("start time:",self.starttime)
        
    def end(self):
        self._oLog.Mode = 'safe'
        self.endtime = datetime.datetime.now()
        self._oLog("end time:",self.endtime)
        
    def append(self,data:dict):
        #save to experiment log excel file
        self.oParentStudy.oExprLog.append(data, self.expr_key)
        #save to .expr file
        expr_record = dict.fromkeys(ExperimentKeys)
        expr_record[EXPR_LOG_FILE_PRIMARY_KEY] = self.expr_key
        expr_record['startTime'] = str(self.starttime)
        expr_record['endTime'] = str(self.endtime)
        if self.keysIncludedInStudyFile:
            for i in self.keysIncludedInStudyFile:
                expr_record[i] = data[i]
        self.oParentStudy.doc['experiment_list'].append(expr_record)
        #save to .study file
        self.oParentStudy.save()
        self._oLog('CExpr Append Finish')
    
    @property
    def oLog(self):
        return self._oLog
    
    def __enter__(self):
        self.start()
        return self.oLog
    
    def __exit__(self,*args):
        self.end()
        #append result only when exits successfully
        if all([i is None for i in args]):
            self.append(self.dataToAppend)
        else:
            self.oLog.t(args[1])
        
        

class CStudy:
    
    def __new__(cls,*args,**kwargs):
        if isinstance(args[0],str):
            return super().__new__(cls)
        elif isinstance(args[0],dict):
            return super().__new__(_CStudy_EasyConfig)
        else:
            raise
    
    def __init__(self,studyHostPath:str,studyName:str,exprLogKeys:list = [],keyNFuncForBest:tuple = None, studyShortcut = None):
        studyHostPath = siDM.CPath(studyHostPath)
        studyPath = studyHostPath / studyName
        siDM.checkFolder(studyPath)
        self.studyPath = studyPath
        self.keyNFuncForBest = keyNFuncForBest
        assert keyNFuncForBest[0] in exprLogKeys or keyNFuncForBest is None
        self.shortcut = studyShortcut
        if not siDM.checkExists(studyPath / STUDY_FILE_NAME):
            print("required file: .study doesn't exist, create a new one? (y/n)")
            #a = input()
            a = 'y'
            if a.lower() == 'y':
                doc = dict()
                for i in StudyKeys:
                    doc[i] = ''
                doc['config'] = dict()
                doc['config']['study_name'] = studyName
                doc['experiment_list'] = []
                self._doc = doc
                self._oExprLog = CStudyExprLogger(self,exprLogKeys, studyPath / studyName + EXPR_LOG_FILE_NAME)
                # self._oExprLog.save()
                # siIO.saveDictJson(studyPath / '.study', doc) #keep as the last line of this code block
        else:
            try:
                self._doc = siIO.loadJson(studyPath / '.study')
                self._oExprLog = CStudyExprLogger(self,exprLogKeys, studyPath / studyName + EXPR_LOG_FILE_NAME)
            except:
                raise
            
    def setStudyName(self,name:str):
        self._doc['study_name'] = name
            
    def getNewExprIndex(self):
        if len(self._oExprLog.df[EXPR_LOG_FILE_PRIMARY_KEY]) == 0:
            return 0
        else:
            out = max(self._oExprLog.df[EXPR_LOG_FILE_PRIMARY_KEY])
            return out + 1
    
    def summary(self):
        if self.keyNFuncForBest:
            key = self.keyNFuncForBest[0]
            func = self.keyNFuncForBest[1]
            tarValue = func(list(self.oExprLog.df[key]))
            df = self.oExprLog.df
            tempKey = EXPR_LOG_FILE_PRIMARY_KEY
            self.doc["best_experiment_indices"] = list(df[tempKey][df[key] == tarValue])
            self.doc['config']["best_experiment_metrics"] = key
    
    def save(self):
        self.summary()
        siIO.saveDictJson(self.studyPath / STUDY_FILE_NAME, self._doc)
        self._oExprLog.save()
    
    @property
    def oExprLog(self):
        return self._oExprLog
    
    def newExpr(self,dataToAppend:dict,keysIncludedInStudyFile:list = None):
        assert isinstance(keysIncludedInStudyFile, list)
        assert self.keyNFuncForBest[0] in dataToAppend
        # if self.keyNFuncForBest:
        #     if keysIncludedInStudyFile is None:
        #         keysIncludedInStudyFile = [self.keyNFuncForBest[0]]
        #     else:
        #         keysIncludedInStudyFile.append(self.keyNFuncForBest[0])
        #         keysIncludedInStudyFile = list(set(keysIncludedInStudyFile))
        if keysIncludedInStudyFile is None:
            keysIncludedInStudyFile = list(dataToAppend.keys())
            if self.keyNFuncForBest:
                keysIncludedInStudyFile.append(self.keyNFuncForBest[0])
        else:
            if self.keyNFuncForBest:
                keysIncludedInStudyFile.append(self.keyNFuncForBest[0])
        keysIncludedInStudyFile = list(set(keysIncludedInStudyFile))
        return CExpr(self, self.getNewExprIndex(),dataToAppend,keysIncludedInStudyFile)
    
    @property
    def doc(self):
        return self._doc
    
    @property
    def bestExprIdx(self):
        # self.summary()
        idx = self.doc['best_experiment_indices'][-1]
        return (self.shortcut, idx)

class _CStudy_EasyConfig(CStudy):

    def __init__(self,pathDict:dict,*args,**kwargs):
        super().__init__(pathDict['root'], pathDict['tag'], *args,studyShortcut= pathDict['shortcut'],**kwargs)
        

    
def studySummary(motherFolder,keywords = [''],onlyBestKey = None,excludes = []):
    excludeString = lambda excludes: '_exclude-'+ '_'.join(excludes) if len(excludes) > 0 else '' 
    bestString = lambda onlyBestKey: '_best-' + str(onlyBestKey) if onlyBestKey else ''
    name = motherFolder + '/' + '_'.join(keywords) + excludeString(excludes) + bestString(onlyBestKey) + '_study_summary.xlsx'
    print(name)
    subfolders = siDM.getSubFolderName(motherFolder)
    oExpr = CStudySummaryExprLogger(name)
    for subFolder in subfolders:
        if not all([i in subFolder for i in keywords]):
            continue
        if len(excludes) >0:
            if any([i in subFolder for i in excludes]):
                continue
        print(f'{motherFolder}/{subFolder}')
        filePath = siDM.getFileList(f'{motherFolder}/{subFolder}','xlsx')[0]
#        print(filePath)
        oExprStudy = CExprFile(filePath)
        df = oExprStudy.df
        if onlyBestKey:
            idx = df[onlyBestKey].idxmax()
            df = df.iloc[idx]
        if onlyBestKey:
            df[EXPR_SUM_FILE_PRIMARY_KEY] = subFolder
        else:
            df[EXPR_SUM_FILE_PRIMARY_KEY] = [subFolder] * len(df)
        oExpr.append(df)
    return oExpr
        
    
    
                
