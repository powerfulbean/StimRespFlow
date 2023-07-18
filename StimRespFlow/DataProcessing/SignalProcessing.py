# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:56:49 2019

@author: sam jin dou
"""
from ..outsideLibInterfaces import _OutsideLibFilter, _OutsideLibTransform
import numpy as np


class CPreprocess:
    
    def __init__(self):
        self._outLibFilter = _OutsideLibFilter()
        self._outLibTransform = _OutsideLibTransform()
    
    def lowPass(self,x,lowpass_hz,order,srate):
        nyq = srate/2.0
        lowpass = lowpass_hz / nyq
        ans = self._outLibFilter.lowPassHannWinFIR(x,lowpass,order+1)
        return ans
    
    def highPass(self,x,highpass_hz, order, srate):
        nyq = srate/2.0
        highpass = highpass_hz / nyq
        ans = self._outLibFilter.highPassHannWinFIR(x,highpass,order+1)
        return ans
        
    def bandPass(self,x,low_hz,high_hz,lowOrder,highOrder, srate):
        nyq = srate/2.0
        lowpass = high_hz / nyq
        highpass = low_hz / nyq
        lowpass_numtaps = highOrder + 1
        highpass_numtaps = lowOrder + 1
        
        ans = self._outLibFilter.bandPassHannWinFIR(
                x, lowpass, highpass,  lowpass_numtaps, highpass_numtaps)
        
        return ans
    
    def getEnvelope(self,x):
        L = len(x)
        while(L%2 != 0):
            L+=1
        print("SigProc: getEnvelope input len",L)
        trans = self._outLibTransform.hilbertTransform(x,N = L) # this funtion can be really slow when the len(x) is a prime
#        print("Transform finish")
        ans = np.abs(trans)
        return ans

def plotFrequencySpectrum(data,srate):
    
    from scipy.fftpack import rfft, irfft, rfftfreq
    time   = np.linspace(0,len(data)/srate,len(data))
    signal = data
    
    f_signal = rfft(signal)
    W = rfftfreq(signal.size, d=time[1]-time[0]) # shouldn't use fftreq (use fftreq will lead to wrong frequency bins)
 
        
    import pylab as plt
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.subplot(2,1,2)
    plt.plot(W,f_signal)
    plt.show()

import scipy
from matplotlib import pyplot as plt
def plotFrequencySpectrumHeatMap(datas, fs):
    nperseg = 1024
    flist,Ss = [],[]
    datas = np.array(datas)
    if datas.ndim == 1:
        datas = datas.reshape(-1,1)
    mat = np.zeros((datas.shape[1], nperseg // 2 + 1))
    for idx,data in enumerate(datas.T):
        (f, S)= scipy.signal.welch(data, fs, nperseg = nperseg, scaling = 'spectrum')
        # print(f)
        mat[idx,:] = S
        flist.append(f)
    assert all([np.array_equal(flist[0], flist[i]) for i in range(1,len(flist))])
    
    fig, ax = plt.subplots()
    ax.imshow(mat, aspect='auto', extent = [0, fs//2, datas.shape[1],0], interpolation="none")
    ax.set_xlabel('frequency / HZ')
    ax.set_ylabel('channels')
    return fig

def audioStimPreprocessing(stimuli,filterHigh,epochLen_s,downSample):
#    if(stimuli == None):
#        stiMain = [0] * int(epochLen_s * downSample)
#        stiMain = np.array(stiMain)
#    else:
    from scipy.signal import resample_poly
    oPre = CPreprocess()
#    print('lowpass')
    stimuliMain = oPre.lowPass(stimuli, filterHigh,900,len(stimuli)/epochLen_s) # !!!!because for now the len of epoch is 60 s, so the srate is len(stimuli)
#    print('lowPass finish')
    outLibTrans = _OutsideLibTransform()
#    print('resample')
#    
    oriSrate = round(len(stimuli)/epochLen_s)
    print(int((oriSrate/4)/downSample))
    stiMainDownSample = resample_poly(stimuliMain,1,int((oriSrate/4)/downSample)) 
#    print(int((oriSrate/4)/downSample))
    #it reauires that the down factor must be integer
    stiMainDownSample = outLibTrans.resample(stiMainDownSample,epochLen_s,downSample)#!!!! now is 60 s
    #Cal Envelope
#    print('envelope')
    stiMain = oPre.getEnvelope(stiMainDownSample)
    return stiMain 
        
def MNECutInFreqBands(data,bands:list,**args):
    ansList = list()
    for idx in range(len(bands)-1):
        highpass = bands[idx]
        lowpass = bands[idx+1]
        temp = data.filter(highpass,lowpass,**args)
        ansList.append(np.expand_dims(temp.get_data(),1))
    ans = np.concatenate(ansList,axis=1)
    return ans
    
def resampleAug(x,oriSrate,newSrate,tarSampleNum):
    from scipy.signal import resample_poly, resample
    x1 = resample_poly(x,newSrate * 4, oriSrate)
    x1 = resample(x1, tarSampleNum)
    return x1

def resample(x,tarSampleNum):
    from scipy.signal import resample
    x1 = resample(x, tarSampleNum)
    return x1

def downsample(x,stepQ,nSteps):
    from scipy.signal import decimate
    for i in range(nSteps):
        x = decimate(x,stepQ)
    return x

def butter_lowpass_filter(data, cutoff, fs, order):
    from scipy.signal import butter, filtfilt
    nyq = fs/2.0
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def getEnvelope(data):
    from scipy.signal import hilbert
    transSig = hilbert(data)
    return np.abs(transSig)

def zscore(data):
    from scipy.stats import zscore
    return zscore(data)
        
        
        
    