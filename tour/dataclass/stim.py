from typing import Dict

from ..backend import is_tensor, np, torch, Array


def to_impulses(x:Array, timeinfo:Array, f:float, padding_s:float = 0):
    '''
    # align the vectors into impulses with specific sampling rate 
    '''
    if is_tensor(x):
        assert is_tensor(timeinfo)
    else:
        assert not is_tensor(timeinfo)
    startTimes = timeinfo[0]
    endTimes = timeinfo[1]
    secLen = endTimes[-1] + padding_s
    nDim = x.shape[0]
    if is_tensor(x):
        nLen = torch.ceil(secLen * f).long()
        out = torch.zeros((nDim, nLen), dtype=x.dtype)
        timeIndices = torch.round(startTimes * f).long()
    else:
        nLen = np.ceil( secLen * f).astype(int)
        out = np.zeros((nDim, nLen), dtype=x.dtype)
        timeIndices = np.round(startTimes * f).astype(int)
    out[:,timeIndices] = x
    return out

def dictTensor_to(x:Dict[str, Array], device):
    output = {
        k:v.to(device) if is_tensor(v) else v for k,v in x.items()
    }
    return output
