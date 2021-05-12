# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:47:52 2021

@author: ShiningStone
"""
import numpy as np

class CStimuliVector():
    '''
    An array-like list
    '''
    
    def __init__(self,nFeat):
        self.nFeat = nFeat
        self._list = list()
        
    def _forceSlice(self,idx):
        assert (type(idx) == int or type(idx) == slice)
        if type(idx) == int:
            idx = slice(idx,idx+1)
        return idx
        
    def _idxTrans(self,idx):
        listSlice = None
        featSlice = None 
        if not (type(idx) == int or type(idx) == slice):
            assert (len(idx) <= 2 and len(idx) >0)
            featSlice = self._forceSlice(idx[0])
            listSlice = self._forceSlice(idx[1])
        else:
            assert (type(idx) == int or type(idx) == slice)
            featSlice = self._forceSlice(idx)
            listSlice = slice(None)
        return featSlice,listSlice
    
    def _idxTransForSetItem(self,idx):
        listSlice = None
        featSlice = None 
        if not (type(idx) == int or type(idx) == slice):
            assert (len(idx) <= 2 and len(idx) >0)
            featSlice = self._forceSlice(idx[0])
            listSlice = idx[1]
        else:
            assert (type(idx) == int or type(idx) == slice)
            featSlice = self._forceSlice(idx)
            listSlice = slice(None)
        return featSlice,listSlice
    
    @property
    def _flagItemIsScalar(self) -> bool:
        return np.isscalar(self._list.__getitem__(0))
    
    def __getitem__(self,idx):
        '''
        idx[0] indicats features, idx[1] indicates timestamp
        '''
        if self._list.__len__() == 0:
            return None
        
        featSlice,listSlice = self._idxTrans(idx)
        # idx1IntFlag = False
        # if type(idx[1]) == int:
        #     idx1IntFlag = True
        # out = CLabelDataList()
        if self._flagItemIsScalar:
            assert featSlice.start == 0 and featSlice.stop == 1
            featSlice = slice(None)
        selectedList = self._list.__getitem__(listSlice)
        sample = selectedList[0]
        nNewFeat = len(sample[featSlice]) if not np.isscalar(sample) else 1
        out = CStimuliVector(nNewFeat)
        for elem in selectedList:
            out.append(elem[featSlice])
        return out
    
    def sliceToIndex(self,oSlice:slice):
        start = oSlice.start
        stop  = oSlice.stop
        step  = oSlice.step
        if start == None:
            start = 0
        if stop == None:
            stop = self.shape[1]
        if step == None:
            step = 1
        return list(range(start,stop,step))
    
    def __setitem__(self,idx,value):
        """
        # After consideration, we feel that it is enough to only allow the user 
        # to set the the stimuliObject using idx corresponded to time dimension.
        
        # Because we don't want to make it so complex that also provides manupulation
        # of data inside a stimuliObject.
        
        Parameters
        ----------
        idx : slice or int
            key for setitem
        value : any
            value for setitem

        Returns
        -------
        None.

        """
        featSlice,listIdxOrSlice = self._idxTransForSetItem(idx)
        selected = self._list.__getitem__(listIdxOrSlice)
        if np.isscalar(selected):
            start = featSlice.start
            stop = featSlice.stop
            if not ((start == 0 and stop == 1) or (start == None and stop == None)):
                raise ValueError("the idx/idx[0] should be 0")
            self._list.__setitem__(listIdxOrSlice, value)
        else:
            listIdxOrSlice = self._forceSlice(listIdxOrSlice)
            indexList = self.sliceToIndex(listIdxOrSlice)
            selected = self._list.__getitem__(listIdxOrSlice)
            # print('120',selected)
            for idx,item in enumerate(selected):
                if np.isscalar(item):
                    start = featSlice.start
                    stop = featSlice.stop
                    if not ((start == 0 and stop == 1) or (start == None and stop == None)):
                        raise ValueError("the idx/idx[0] should be 0")
                    self._list.__setitem__(indexList[idx],value[idx][:])
                else:
                    selected[idx][featSlice] = value[idx][:]
        
    def append(self,stimuliObject):
        if np.isscalar(stimuliObject):
            assert self.nFeat == 1
        else:
            assert len(stimuliObject) == self.nFeat
        self._list.append(stimuliObject)
        
    def __len__(self):
        return self.nFeat
    
    @property
    def shape(self):
        return tuple([self.nFeat,self._list.__len__()])
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self._list})"
    
    def __array__(self, dtype=None):
        return np.array(self._list).T
    
    def numpy(self,dtype=None):
        return self.__array__(dtype)

# class CStimuliVector(list):
#     '''
#     An array-like list
#     '''
    
#     def __init__(self,nFeat):
#         list().__init__([])
#         self.nFeat = nFeat
        
#     def _forceSlice(self,idx):
#         assert (type(idx) == int or type(idx) == slice)
#         if type(idx) == int:
#             idx = slice(idx,idx+1)
#         return idx
        
#     def _idxTrans(self,idx):
#         listSlice = None
#         featSlice = None 
#         if not (type(idx) == int or type(idx) == slice):
#             assert (len(idx) <= 2 and len(idx) >0)
#             featSlice = self._forceSlice(idx[0])
#             listSlice = self._forceSlice(idx[1])
#         else:
#             assert (type(idx) == int or type(idx) == slice)
#             featSlice = self._forceSlice(idx)
#             listSlice = slice(None)
#         return featSlice,listSlice
    
#     def _idxTransForSetItem(self,idx):
#         listSlice = None
#         featSlice = None 
#         if not (type(idx) == int or type(idx) == slice):
#             assert (len(idx) <= 2 and len(idx) >0)
#             featSlice = self._forceSlice(idx[0])
#             listSlice = idx[1]
#         else:
#             assert (type(idx) == int or type(idx) == slice)
#             featSlice = self._forceSlice(idx)
#             listSlice = slice(None)
#         return featSlice,listSlice
    
#     @property
#     def _flagItemIsScalar(self) -> bool:
#         return np.isscalar(super().__getitem__(0))
    
#     def __getitem__(self,idx):
#         '''
#         idx[0] indicats features, idx[1] indicates timestamp
#         '''
#         if super().__len__() == 0:
#             return None
        
#         featSlice,listSlice = self._idxTrans(idx)
#         # idx1IntFlag = False
#         # if type(idx[1]) == int:
#         #     idx1IntFlag = True
#         # out = CLabelDataList()
#         if self._flagItemIsScalar:
#             assert featSlice.start == 0 and featSlice.stop == 1
#             featSlice = slice(None)
#         selectedList = super().__getitem__(listSlice)
#         sample = selectedList[0]
#         nNewFeat = len(sample[featSlice]) if not np.isscalar(sample) else 1
#         out = CStimuliVector(nNewFeat)
#         for elem in selectedList:
#             out.append(elem[featSlice])
#         return out
    
#     def sliceToIndex(self,oSlice:slice):
#         start = oSlice.start
#         stop  = oSlice.stop
#         step  = oSlice.step
#         if start == None:
#             start = 0
#         if stop == None:
#             stop = self.shape[1]
#         if step == None:
#             step = 1
#         return list(range(start,stop,step))
    
#     def __setitem__(self,idx,value):
#         """
#         # After consideration, we feel that it is enough to only allow the user 
#         # to set the the stimuliObject using idx corresponded to time dimension.
        
#         # Because we don't want to make it so complex that also provides manupulation
#         # of data inside a stimuliObject.
        
#         Parameters
#         ----------
#         idx : slice or int
#             key for setitem
#         value : any
#             value for setitem

#         Returns
#         -------
#         None.

#         """
#         featSlice,listIdxOrSlice = self._idxTransForSetItem(idx)
#         selected = super().__getitem__(listIdxOrSlice)
#         if np.isscalar(selected):
#             start = featSlice.start
#             stop = featSlice.stop
#             if not ((start == 0 and stop == 1) or (start == None and stop == None)):
#                 raise ValueError("the idx/idx[0] should be 0")
#             super().__setitem__(listIdxOrSlice, value)
#         else:
#             indexList = self.sliceToIndex(listIdxOrSlice)
#             for idx,item in enumerate(selected):
#                 if np.isscalar(item):
#                     super().__setitem__(indexList[idx],value[idx][:])
#                 else:
#                     selected[idx][featSlice] = value[idx][:]
        
#     def append(self,stimuliObject):
#         if np.isscalar(stimuliObject):
#             assert self.nFeat == 1
#         else:
#             assert len(stimuliObject) == self.nFeat
#         super().append(stimuliObject)
        
#     def __len__(self):
#         return self.nFeat
    
#     @property
#     def shape(self):
#         return tuple([self.nFeat,super().__len__()])
    
#     def numpy(self,):
#         return np.array(self).T