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
    plt.plot(W,f_signal)
    plt.show()

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
        
        
         
        
    