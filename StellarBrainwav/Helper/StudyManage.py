# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:15:15 2021

@author: ShiningStone
"""
import datetime

from StellarInfra import DirManage as siDM
from StellarInfra import IO as siIO
from StellarInfra.Logger import CExprLogger,CLog

STUDY_FILE_NAME = '.study'
EXPR_FILE_NAME = '.expr'
EXPR_LOG_FILE_NAME = 'stellarBrainwav_expr.xlsx'
EXPR_LOG_FILE_PRIMARY_KEY = 'expr_index'

StudyKeys = ['study_name','best_experiment_indices','best_experiment_metrics','experiment_list']
ExperimentKeys = [EXPR_LOG_FILE_PRIMARY_KEY,'startTime','endTime']

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
        self._oLog.Mode = 'fast'
        self.starttime = datetime.datetime.now()
        self._oLog("start time:",self.starttime)
        
    def end(self):
        self.endtime = datetime.datetime.now()
        self._oLog("end time:",self.endtime)
        self._oLog.Mode = 'safe'
        
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
        self.append(self.dataToAppend)
        

class CStudy:
    def __init__(self,studyHostPath:str,studyName:str,exprLogKeys:list,keyNFuncForBest:tuple = None):
        studyHostPath = siDM.CPath(studyHostPath)
        studyPath = studyHostPath / studyName
        siDM.checkFolder(studyPath)
        self.studyPath = studyPath
        self.keyNFuncForBest = keyNFuncForBest
        if not siDM.checkExists(studyPath / STUDY_FILE_NAME):
            print("required file: .study doesn't exist, create a new one? (y/n)")
            a = input()
            if a.lower() == 'y':
                doc = dict()
                for i in StudyKeys:
                    doc[i] = ''
                doc['study_name'] = studyName
                doc['experiment_list'] = []
                self._oExprLog = CStudyExprLogger(self,exprLogKeys, studyPath / EXPR_LOG_FILE_NAME)
                self._oExprLog.save()
                siIO.saveDictJson(studyPath / '.study', doc) #keep as the last line of this code block
        try:
            self._doc = siIO.loadJson(studyPath / '.study')
            self._oExprLog = CStudyExprLogger(self,exprLogKeys, studyPath / EXPR_LOG_FILE_NAME)
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
            self.doc["best_experiment_metrics"] = key
    
    def save(self):
        self.summary()
        siIO.saveDictJson(self.studyPath / STUDY_FILE_NAME, self._doc)
        self._oExprLog.save()
    
    @property
    def oExprLog(self):
        return self._oExprLog
    
    def newExpr(self,dataToAppend,keysIncludedInStudyFile:list = None):
        assert isinstance(keysIncludedInStudyFile, list)
        if self.keyNFuncForBest:
            if keysIncludedInStudyFile is None:
                keysIncludedInStudyFile = [self.keyNFuncForBest[0]]
            else:
                keysIncludedInStudyFile.append(self.keyNFuncForBest[0])
                keysIncludedInStudyFile = list(set(keysIncludedInStudyFile))
        return CExpr(self, self.getNewExprIndex(),dataToAppend,keysIncludedInStudyFile)
    
    @property
    def doc(self):
        return self._doc
    
        
    
                
