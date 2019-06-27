import struct 
import numpy as np

    
def loadSound(Dir,Fs=22050):
    import librosa  
    wav,sr = librosa.load(Dir,Fs)
    return wav

def loadBin(Dir):
    f = open(Dir,'rb')
    data=[]
    
    while True:
        b = f.read(8)
        if(len(b)==0):
            break
        else:
            data.append(struct.unpack('d',b)) 
            
    caldata=[float(p[0]) for p in data]
    f.close()
    return caldata

def loadJpg(Dir,i = None, iMax = None):
    from PIL import Image
    return np.array(Image.open(Dir))[:,:,2]
