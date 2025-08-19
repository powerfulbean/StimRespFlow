import torch

def pearsonr_loss(x, y):
    return -Pearsonr(x,y)

def Pearsonr(x,y):
    nObs = len(x)
    sumX = torch.sum(x,0)
    sumY = torch.sum(y,0)
    sdXY = torch.sqrt((torch.sum(x**2,0) - (sumX**2/nObs)) \
                      * (torch.sum(y ** 2, 0) - (sumY ** 2)/nObs))    
    r = (torch.sum(x*y,0) - (sumX * sumY)/nObs) / sdXY
    return r

def BatchPearsonr(pred,y,batchAvg = True):
    result = list()
    for i in range(len(pred)):
        out1 = Pearsonr(pred[i],y[i])
        result.append(out1)
        # print(np.mean(result,0).shape)
    result = torch.stack(result, dim = 0)
    if batchAvg:
        return torch.mean(result, dim = 0)
    else:
        return result
    
