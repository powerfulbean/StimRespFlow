# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:37:01 2021

@author: ShiningStone
"""

from StimRespFlow.outsideLibInterfaces import CIfMNE
import numpy as np
# oTemp = ConfResStudy('Stages.conf')
class CSpatialData:
    '''
    data should be the size of (nChannel,nLen)
    '''
    def __init__(self,chNames,montage,data,adapterFunc = lambda x: x):
        self.chNames = chNames
        self.montage = montage
        data = adapterFunc(data)
        self._data = np.array(data)
            
    def topoplot(self,srate = 1,chanIdx = None,paramsDict={},**kwargs):
        data = self._data
        oMNE = CIfMNE(self.chNames,srate,'eeg',self.montage)
        oWeight = oMNE.getMNEEvoked(data)
        chanMask = None
        if chanIdx is not None:
            chanMask = np.zeros(data.shape,dtype = bool)
            for i in chanIdx:
                chanMask[i] = True
        
        maskParam = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
        linewidth=0, markersize=8)
        fig2 = oWeight.plot_topomap([0],res = 256,sensors=False,
                    cmap='jet',outlines='skirt',time_unit='s',mask = chanMask,mask_params= maskParam,**kwargs)
        fig = fig2
        temp1 = fig.get_axes()
        t1 = temp1[0]
        t1.set_title(paramsDict['title0'] if paramsDict.get('title0') else '')
        t2 = temp1[-1]
        # t2.set_axis_off()
        t2.set_title(paramsDict['title1'] if paramsDict.get('title1') else '')
        
        ticklabels = list()
        t3 = t2.get_yticklabels()
        for i in range(len(t3)):
            print(str(t3[i].get_position()[1] / 1e6))
            ticklabels.append(str(t3[i].get_position()[1] / 1e6))
        t2.set_yticklabels(ticklabels)
        t2.set_axis_on()
        
        return fig2
    
    def condTopoplot(self,condFunc,srate = 1,**kwargs):
        idx = np.where(condFunc(self._data))[0]
        return self.topoplot(srate,idx,**kwargs),idx