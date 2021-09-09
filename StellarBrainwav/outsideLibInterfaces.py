# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:17:35 2019

@author: Sam Jin Dou
"""
#from .DataStruct.DataSet import CDataSet
import numpy as np
import mne

class _OutsideLibFilter:
    
    def __init__(self):
        pass
    
    def _importScipySignal(self):
        import scipy.signal as Signal     
        return Signal
    
    def bandPassHannWinFIR(self,x,lowpass,highpass,lowTaps,highTaps):
        signal = self._importScipySignal()
        import scipy
        version = scipy.__version__
        if(version < '1.3.0'):
            #low pass
            lowpassCo = signal.firwin(lowTaps, lowpass,pass_zero = True, window = "hann")
            #high pass
            highpassCo = signal.firwin(highTaps, highpass,pass_zero = False, window = "hann")  
        else:
            #low pass
            lowpassCo = signal.firwin(lowTaps, lowpass,pass_zero = 'lowpass', window = "hann")
            #high pass
            highpassCo = signal.firwin(highTaps, highpass, pass_zero = 'highpass', window = "hann")   
#        print(highpass,lowpass)
        #perform lowpass filter 
        x1 = signal.convolve(x,lowpassCo,mode = 'same')
        #perform highpass filter
        x2 = signal.convolve(x1,highpassCo, mode = 'same')     
        return x2
    
    def lowPassHannWinFIR(self,x,lowpass,numtaps):
        
        signalLib = self._importScipySignal()
        import scipy
        version = scipy.__version__
        if(version < '1.3.0'):
            lowpassCo = signalLib.firwin(numtaps, lowpass,pass_zero = True, window = "hann")
        else:
            lowpassCo = signalLib.firwin(numtaps, lowpass,pass_zero = 'lowpass', window = "hann")
            
#        print('lowPassHannWinFIR get coefficient')
    
        ans = signalLib.convolve(x, lowpassCo, mode = 'same')
#        print('convolve finish')
        return ans

    def highPassHannWinFIR(self,x,highpass,numtaps):
        
        signalLib = self._importScipySignal()
        import scipy
        version = scipy.__version__
        if(version < '1.3.0'):
            highpassCo = signalLib.firwin(numtaps, highpass,pass_zero = False, window = "hann")
        else:
            highpassCo = signalLib.firwin(numtaps, highpass,pass_zero = 'highpass', window = "hann")
        ans = signalLib.convolve(x, highpassCo, mode = 'same')
        
        return ans
    
    
class _OutsideLibTransform:
    
    def __init__(self):
        pass
    
    def _importScipySignal(self):
        import scipy.signal as Signal
        return Signal
    
    def hilbertTransform(self,signal, N):
        Lib = self._importScipySignal()
        ans = Lib.hilbert(signal,N)
        return ans
    
    def resample(self,x,oriLen_s,targetF):
#        print("start resample")
        n = round(oriLen_s * targetF)
        Lib = self._importScipySignal()
        ans = Lib.resample(x,n)
        return ans
    
    def getRFFTweight(self,x,fs):
        from scipy.fftpack import rfft, irfft, rfftfreq
        bins = rfftfreq(x.size, d=1/fs)
        W = rfft(x)
        return W,bins
  
class _OutsideLibML:
    
    def __init_(self):
        pass
    
    def _importSklearnDecompostion(self):
        import sklearn.decomposition as decom
        return decom
    
    def FastICA(self,x,num_components):
        decomMod = self._importSklearnDecompostion()     
        ica = decomMod.FastICA(num_components)
        s = ica.fit_transform(x)
        return s
        
        
class _OutsideLibTime:

    def __init__(self):
        pass
    
    def _importDatetime(self):
        import datetime as date
        return date
    
class _OutsideLibIO:
    
    def __init__(self):
        pass
    
    def _importPydub(self):
        import pydub as pydub
        return pydub
    
    def _importJson(self):
        import json as Json
        return Json
    
    def _importScipyIO(self):
        import scipy.io as sio
        return sio
    
    def writeMATFile(self,Dict, FileName):
        sio = self._importScipyIO()
        sio.savemat(FileName, Dict)
        
    def loadMATFile(self,FileName):
        sio = self._importScipyIO()
        bdict = sio.loadmat(FileName)
        return bdict
    
    def getMonoChannelData(self,file):
        pydub = self._importPydub()
        sound = pydub.AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        frameNum = sound.frame_count()
#        print(frameNum)
        channelNum = sound.channels
#        print(channelNum)
        data = sound.raw_data
        return frameNum, channelNum, data, len(sound)/1000
    
    
class CIfMNE:

    def __init__(self,chNames=None,srate=None,chTypes=None,montage = None,oLog = None):
        self.LibMNE = self._importMNE()
        if (chNames is None) or (srate is None):
            self.info = None
        else:
            self.info = self.LibMNE.create_info(chNames, srate,ch_types = chTypes)
        # if(montage != None):
        self.Montage = montage
        self.oLog = oLog
    
    def _importMNE(self):
        import mne as MNE
        return MNE
    
    def getMNERaw(self,data,channelsInfo=None,srate=None,chTypes:list=None) -> mne.io.RawArray:
        oRaw = None
        if self.info is None or ((channelsInfo is not None) and (chTypes is not None)):
            info = self.LibMNE.create_info(channelsInfo, srate,ch_types = chTypes)
            oRaw = self.LibMNE.io.RawArray(data,info)
        else:
            oRaw = self.LibMNE.io.RawArray(data,self.info)
        if self.Montage is not None:
            try:
                oRaw.set_montage(self.Montage)
            except:
                print('failed to set montage')
        return oRaw

    def getMNEEvoked(self,data,channelsInfo=None,srate=None,chTypes:list=None):
        oRaw = None
        if self.info is None :
            info = self.LibMNE.create_info(channelsInfo, srate,ch_types = chTypes)
            oRaw = self.LibMNE.EvokedArray(data,info)
        else:
            oRaw = self.LibMNE.EvokedArray(data,self.info)
        if self.Montage is not None:
            oRaw.set_montage(self.Montage)
        return oRaw             
    
    def CDataSetToEpochs(self,oDataSet,eventIdList,eventIdDict):
        if(len(eventIdList) != len(oDataSet.dataRecordList)):
            raise ValueError()
        epochs_data = list()
        events = list()
        cnt = 0
        for idx, dataRecord in enumerate(oDataSet.dataRecordList):
            data = dataRecord.data
            events.append([cnt,0,eventIdList[idx]])
            if(data.shape != oDataSet.dataRecordList[0].data.shape):
                shape1 = data.shape
                shape = oDataSet.dataRecordList[0].data.shape
                data = data[0:shape[0],0:shape[1]]
                if(self.oLog!=None):
                    self.oLog.safeRecordTime(str(idx) + "'s shape" + str(shape1)  + " is different")
            cnt += data.shape[1]
            data = np.expand_dims(data,0)
            epochs_data.append(data)
            
        epochs_data = np.concatenate(epochs_data,axis = 0)
        events = np.array(events)
        epochs = self.LibMNE.EpochsArray(epochs_data, info=self.info, events=events,
                         event_id=eventIdDict)
        
        epochs.set_montage(self.Montage)
        
        return epochs
        
        
class CIfSklearn:
    
    def __init__(self):
        self._sk = None
        self.get()
    
    @property
    def Lib(self):
        return self._sk
    
    @property
    def metricsLib(self):
        from sklearn import metrics
        return metrics
    
    def get(self):
        import sklearn as sk
        self._sk = sk
        return sk
    
    def preprocessing(self):
        import sklearn.preprocessing as pre
        return pre
    
    def model_selection(self):
        import sklearn.model_selection as modsel
        return modsel

class CIfScipy:
    
    def __init__(self):
        pass
    
    def stats(self):
        from scipy import stats
        return stats
    
class CIfKeras:
    
    def __init__(self):
        pass
    
    def keras(self):
        import keras
        return keras