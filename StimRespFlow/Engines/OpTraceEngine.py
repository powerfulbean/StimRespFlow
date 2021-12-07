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
        return list(set(self._opRecords))
    
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