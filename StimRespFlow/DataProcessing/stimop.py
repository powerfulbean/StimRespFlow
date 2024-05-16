import numpy as np
import scipy

class SingleDataTransform:

    def __init__(self,):
        pass

    def process(self, x):
        pass

    def __call__(self, x):
        return self.process(x)
    
class getAcousticOnset(SingleDataTransform):

    def __init__(self, input_fs, zscore = True, threshold = 1e-4):
        self.zscore = zscore
        self.threshold = threshold
        self.input_fs = input_fs

    def process(self, envelope):
        acoustic_on = np.diff(envelope, prepend = 0)
        acoustic_on = np.maximum(acoustic_on,0)
        acoustic_on[acoustic_on < self.threshold] = 0
        value = acoustic_on[acoustic_on > 0]
        idx = np.where(acoustic_on > 0)[1]
        if self.zscore:
            assert value.ndim == 1
            assert acoustic_on.shape[0] == 1
            value = scipy.stats.zscore(value)
            # acoustic_on[acoustic_on > 0] = value
            acoustic_on = scipy.stats.zscore(acoustic_on, axis = -1)
        tIntvl = idx / self.input_fs
        value = value[None,:]
        tIntvl = tIntvl[None,:]
        return {
            f'acoustic_onset_fs{self.input_fs}': acoustic_on, 
            'acoustic_onset':{
                'values': value,
                'timestamps': tIntvl
            },
        }

def featDictProcess(func):
    def wrapper(stimDict, src, *tars):
        for k in stimDict:
            result = func(stimDict[k][src])
            if len(tars) == 1:
                result = [result]
            for idx,tar in enumerate(tars):
                stimDict[k][tar] = result[idx] 
        return stimDict
    return wrapper


def selectFeatByName(data, featlist, tarFeats):
    #data: nRepition * nRuns * (nSamples, nFeat)
    tarFeatIdx = np.array([featlist.index(f) for f in tarFeats])
    data = list(
        map(lambda d: [d_[:, tarFeatIdx] for d_ in d], data)
    )
    return data