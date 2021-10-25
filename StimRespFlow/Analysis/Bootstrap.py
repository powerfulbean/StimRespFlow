# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:15:05 2021

@author: ShiningStone
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def non_parametric_bootstrap(x, f, niter=500, **kwargs):
    """
    modified from https://wormlabcaltech.github.io/mprsq/stats_tutorial/nonparametric_bootstrapping.html
    Params:
    x - data (numpy arrays)
    f - test function to calculate
    niter - number of iterations to run
    """
    statistic = np.zeros(niter)
    rng = np.random.default_rng(42)
    for i in range(niter):
        # simulate x
        # indices = np.random.randint(0, len(x), len(x))
        indices = rng.integers(0,len(x),len(x))
        X = x[indices]       
        statistic[i] = f(X, **kwargs)
    return statistic



def getMean_CI(sampled,alphaSS = 0.025,fig = False):
    '''
    modified from https://wormlabcaltech.github.io/mprsq/stats_tutorial/nonparametric_bootstrapping.html
    '''
    sampled = np.sort(sampled)
    mean = sampled.mean()
    message = "Mean = {0:.2g}; CI = [{1:.2g}, {2:.2g};]"
    left = int(np.floor(alphaSS*len(sampled)))
    right = int(np.floor((1-alphaSS)*len(sampled)))
    print(left,right)
    # pvalue = len(sampled[sampled < 0])/len(sampled)
    if fig:
        sns.distplot(sampled)
    print(message.format(mean, sampled[left], sampled[right])) #,pvalue
    return mean,(sampled[left], sampled[right])#,pvalue

def test_null(x, y, statistic, iters=1000):
    #from https://wormlabcaltech.github.io/mprsq/stats_tutorial/nonparametric_bootstrapping.html
    """
    Given two datasets, test a null hypothesis using a permutation test for a given statistic.
    
    Params:
    x, y -- ndarrays, the data
    statistic -- a function of x and y
    iters -- number of times to bootstrap
    
    Ouput:
    a numpy array containing the bootstrapped statistic
    """
    #!!!this method may be wrong because our data is paired (subject and trial paired)
    def permute(x, y):
        """Given two datasets, return randomly shuffled versions of them"""
        # concatenate the data
        new = np.concatenate([x, y])
        # shuffle the data
        np.random.shuffle(new)
        # return the permuted data sets:
        return new[:len(x)], new[len(x):]
    
    return np.array([statistic(*permute(x, y)) for _ in range(iters)])

def nonPairedDiffTest(baseData, compData,func,iters = 10**4,plotFig = False):
    '''
    https://wormlabcaltech.github.io/mprsq/stats_tutorial/nonparametric_bootstrapping.html#Generating-Synthetic-Data
    '''
    diffSamples = test_null(baseData, compData, func,iters=iters)
    diffObs = compData.mean() - baseData.mean()
    if plotFig:
        plt.figure()
        sns.distplot(diffSamples)
        plt.axvline(diffObs, color='red',label='Observed Difference')
        plt.title('Bootstrapped Difference in Sample Means')
        plt.xlabel('Difference in Means')
        plt.ylabel('Density')
        plt.legend()
    pValue = len(diffSamples[diffSamples > diffObs])/len(diffSamples)
    print('The p-value for these samples is {0:.2g}'.format(pValue))
    return diffObs,pValue

def pairedDiffTest(baseData, compData, func,alphaSS,niter=10**4, fig = True):
    diffPaired = compData - baseData
    samples = non_parametric_bootstrap(diffPaired, func,niter)
    temp = getMean_CI(samples,alphaSS,fig)
    return temp

#hirachical bootstrap 
# for each subject - trial level
# for each trial - subject level 