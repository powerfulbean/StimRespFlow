import struct 
import numpy as np

    
def loadSound(self,Dir,Fs=22050):
    import librosa  
    wav,sr = librosa.load(Dir,Fs)
    return wav

def loadBin(self,Dir,Num = 256):
    f = open(Dir,'rb')
    data=[]
    for k in range(Num):
        b=f.read(8)
        data.append(struct.unpack('d',b))       
    caldata=[float(p[0]) for p in data]
    f.close()
    return caldata

def loadJpg(self,Dir,i = None, iMax = None):
    from PIL import Image
    return np.array(Image.open(Dir))[:,:,2]
