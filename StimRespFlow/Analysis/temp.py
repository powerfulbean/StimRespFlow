# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 00:27:54 2022

@author: ShiningStone
"""

from StellarInfra import IO as siIO
from StimRespFlow import CFlowDict
from scipy.stats import shapiro
import numpy as np

temp = siIO.loadObject(r'D:\OneDrive\Dataset\WordVectors\OldManAndTheSea\PermMaskForOnlyLookAhead_genVec-sentWise~SentByLine\SRL\Top4\35_33_finetune4layer_06142021.vec')
vecList = []
for i in temp:
    vecList.extend(temp[i]['vec'])
    
vecArr = np.stack(vecList,axis=0)

result = shapiro(vecArr)