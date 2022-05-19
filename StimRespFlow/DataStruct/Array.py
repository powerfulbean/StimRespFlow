# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:47:52 2021

@author: ShiningStone
"""
import numpy as np
import copy
import warnings
class CStimuliVectors():
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
        assert (type(idx) == int or type(idx) == slice or len(idx) == 2)
        if (type(idx) is int or type(idx) is slice):  
            featSlice = idx
            listSlice = slice(None)
        else:
            featSlice = idx[0]
            listSlice = idx[1]
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
        
        featSlice,listSlice = self._idxTransForSetItem(idx)
        # idx1IntFlag = False
        # if type(idx[1]) == int:
        #     idx1IntFlag = True
        # out = CLabelDataList()
        if self._flagItemIsScalar:
            assert featSlice.start == 0 and featSlice.stop == 1
            featSlice = slice(None)
        selected = self._list.__getitem__(listSlice)
        if type(selected) == list:
            sample = selected[0]
            nNewFeat = len(sample[featSlice]) if not np.isscalar(sample) else 1
            out = CStimuliVectors(nNewFeat)
            for elem in selected:
                out.append(elem[featSlice])
        else:
            out = selected
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
        featIdx,listIdx = self._idxTrans(idx)
        selected = self._list.__getitem__(listIdx)
        if type(listIdx) is int:
            if np.isscalar(selected):
                assert np.isscalar(value)
                self._list.__setitem__(listIdx, value)
            else:
                self._list.__getitem__(listIdx)[featIdx] = value[:]
        else:
            indexList = self.sliceToIndex(listIdx)
            selected = self._list.__getitem__(listIdx)
            assert len(value) == len(indexList)
            for idx,item in enumerate(selected):
                if np.isscalar(item):
                    assert np.isscalar(value[idx])
                    self._list.__setitem__(indexList[idx], value[idx])
                else:
                    selected[idx][featIdx] = value[idx][:]
                    
    def replace(self,idx,vec):
        nNewFeat = len(vec)
        self._list[idx][0:nNewFeat] = vec
        self._list[idx] = self._list[idx][0:nNewFeat]
        # pass
                    
    # def __setitem__(self,idx,value):
    #     """
    #     # After consideration, we feel that it is enough to only allow the user 
    #     # to set the the stimuliObject using idx corresponded to time dimension.
        
    #     # Because we don't want to make it so complex that also provides manupulation
    #     # of data inside a stimuliObject.
        
    #     Parameters
    #     ----------
    #     idx : slice or int
    #         key for setitem
    #     value : any
    #         value for setitem

    #     Returns
    #     -------
    #     None.

    #     """
    #     featSlice,listIdxOrSlice = self._idxTransForSetItem(idx)
    #     selected = self._list.__getitem__(listIdxOrSlice)
    #     if np.isscalar(selected):
    #         start = featSlice.start
    #         stop = featSlice.stop
    #         if not ((start == 0 and stop == 1) or (start == None and stop == None)):
    #             raise ValueError("the idx/idx[0] should be 0")
    #         self._list.__setitem__(listIdxOrSlice, value)
    #     else:
    #         listIdxOrSlice = self._forceSlice(listIdxOrSlice)
    #         indexList = self.sliceToIndex(listIdxOrSlice)
    #         selected = self._list.__getitem__(listIdxOrSlice)
    #         # print('120',selected)
    #         assert len(value) == len(indexList)
    #         for idx,item in enumerate(selected):
    #             if np.isscalar(item):
    #                 start = featSlice.start
    #                 stop = featSlice.stop
    #                 if not ((start == 0 and stop == 1) or (start == None and stop == None)):
    #                     raise ValueError("the idx/idx[0] should be 0")
    #                 self._list.__setitem__(indexList[idx],value[idx][:])
    #             else:
    #                 selected[idx][featSlice] = value[idx][:]
                    
    def __call__(self,tIdx):
        """
        Get the stimulus object at the specific tIdx
        """
        assert type(tIdx) == int
        return self._list[tIdx]
    
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
    
    def __iter__(self):
        return CStimuliVectors.Iterator(self)
    
    def numpy(self,dtype=None):
        return self.__array__(dtype)
    
    def clear(self,):
        self._list.clear()
        
    def copy(self):
        new = copy.deepcopy(self)
        new._list = self._list.copy()
        return new

    class Iterator:
       ''' Iterator class '''
       def __init__(self, vector):
           self._vector = vector
           self._index = 0
       def __next__(self):
           ''''Returns the next value from team object's lists '''
           if self._index < len(self._vector._list):
               out = self._vector._list[self._index]
               self._index += 1
               return out
           raise StopIteration
       
class CStimulusVector(np.ndarray):
    
    def __new__(cls,shape,*args,**kwargs):
        obj = super().__new__(cls,shape,*args,**kwargs)
        addedAttr:dict = cls.configAttr()
        for i in addedAttr:
           setattr(obj, i, addedAttr[i])
        obj.attrDict = addedAttr
        obj.info = None
        return obj
    
    @classmethod
    def configAttr()->dict:
        return {'name':'CStimuliVector'}

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'name', None)
        addedAttr:dict = obj.__class__.configAttr()
        for i in addedAttr:
           setattr(self, i, getattr(obj,i)) 
        setattr(self, 'attrDict',addedAttr)  
        
    def __len__(self):
        return self.shape[0]
    
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.info,self.attrDict)
        for i in self.attrDict:
           val = getattr(self, i, getattr(self,i))
           self.attrDict[i] = val
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.info = state[-2]  # Set the info attribute
        addedAttr = state[-1]
        for i in addedAttr:
           setattr(self, i, addedAttr[i]) 
        setattr(self, 'attrDict',addedAttr)  
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-2])

class CWaveArray(np.ndarray):
    def __new__(cls,nChan,arrLike = None ,*args,**kwargs):
        if arrLike is None:
            obj = super().__new__(cls,(nChan,0),*args,**kwargs)
        else:
            arr = np.array(arrLike)
            arr = arr.view(CWaveArray)
            if arr.shape[0] != nChan:
                raise ValueError(f"The channel number of input array should be {nChan}")
            else:
                obj = arr
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # print(self.shape,obj.shape,self.nChan)
        # print(obj.shape[0] == self.nChan)
        if obj.shape[0] != self.nChan:
            raise ValueError("The channel numbers of two ndarray should be the same")
        # see InfoArray.__array_finalize__ for comments
        # super().__array_finalize__(obj)
        
    def __len__(self):
        return self.shape[1]
    
    @property
    def nLen(self):
        return self.shape[1]
    
    @property
    def nChan(self):
        return self.shape[0]
    
    def resize(self, new_shape):
        warnings.warn(f"The function 'resize' for {str(self.__class__)} is forbidden")
        return self
    
    def sort(self,*args,**kwargs):
        warnings.warn(f"The function 'sort' for {str(self.__class__)} is forbidden")
        return self
    
    
    
