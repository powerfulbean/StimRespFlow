from typing import Dict

from ..backend import is_tensor, np, torch, Array

def combine_stim_dict(*stimdicts):
    data = {}
    for stimdict in stimdicts:
        for ID in stimdict:
            if ID not in data:
                data[ID] = {}
            for feat in stimdict[ID]:
                assert feat not in data[ID]
                data[ID][feat] = stimdict[ID][feat]
    return data

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

def exclude_words(t_feat_dict:Dict, excluded_words):
    new_x = []
    new_timeinfo = []
    new_word = []
    for idx, w in enumerate(t_feat_dict['tag']):
        if w not in excluded_words:
            new_x.append(t_feat_dict['x'][:, idx])
            new_timeinfo.append(t_feat_dict['timeinfo'][:, idx])
            new_word.append(w)
    new_x = np.stack(new_x, axis = -1)
    new_timeinfo = np.stack(new_timeinfo, axis = -1)
    t_new = {
        'x':new_x,
        'timeinfo': new_timeinfo,
        'tag': new_word
    }
    return t_new