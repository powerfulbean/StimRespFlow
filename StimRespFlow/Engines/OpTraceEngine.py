# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:52:04 2021

@author: Jin Dou
"""

class MixinTraceable:
    
    def __init__(self):
        self._opRecords = list()
        
    @property
    def opRecords(self):
        output = []
        output.extend(list(set(self._opRecords)))
        for k,v in self.__dict__.items():
            if isinstance(v, MixinTraceable):
                output.extend(v.opRecords)
        if getattr(self, 'children',None) is not None:
            for v in self.children():
                opRecords = getattr(v, 'opRecords',None)
                if opRecords is not None:
                    output.extend(opRecords)
        output.sort()
        return output
    
    @property
    def opRecordsData(self):
        output = dict()
        records = self.opRecords
        for record in records:
            params = record.split('-')
            grpName = params[0]
            if output.get(grpName,None) is None:
                output[grpName] = dict()
            for i in range(1,len(params)):
                kv = params[i].split('~')
                output[grpName][kv[0]] = kv[1]
        return output
    
    def addOp(self,doc:str):
        doc = str(doc)
        if self.__dict__.get('_opRecords') is None:
            raise AttributeError("Can't add new op records before MixinTraceable.__init__() call")
            
        self._opRecords.append(doc)
        
    def tracedOp(doc:str,kwToRecord:list = []):
        #this is decorator to config doc string for decorated operation
        def _tracedOpWithDoc(func):
            def _wrapper(self,*args,**kwargs):
                # print(self)
                newDoc = doc
                if not isinstance(self, MixinTraceable):
                    raise TypeError("the decorated method should belong to a 'MixinTraceable' type object")
                for kw in kwToRecord:
                    newDoc += f'-{kw}~{kwargs[kw]}'
                self.addOp(newDoc)
                return func(self,*args,**kwargs)
            return _wrapper
        return _tracedOpWithDoc