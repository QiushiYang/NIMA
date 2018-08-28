#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:01:10 2018

@author: qsyang
"""

import numpy as np
import torch
import math
from scipy.stats import pearsonr
from scipy.stats import spearmanr
    
def LCC(x,y):
    '''
    Calculate the LCC of the two 1D tensors
    Args:
        x,y are 1D tensors
    '''
    x = np.array(x)
    y = np.array(y)
    return pearsonr(x,y)[0]

def LCC_Mean(x,y):
    '''
    Calculate LCC of the row-mean of 2D tensor : x,y
    Args:
        x,y are 2D tensors with the shape: [testset_size, 10]
    '''
    x_mean = x.mean(dim=1)
    y_mean = y.mean(dim=1) 
    return LCC(x_mean,y_mean)

def LCC_Std(x,y):
    '''
    Calculate LCC of the row-std of 2D tensor : x,y
    Args:
        x,y are 2D tensors with the shape: [testset_size, 10]
    '''
    x_std = x.std(dim=1)
    y_std = y.std(dim=1)
    return LCC(x_std,y_std)

def SRCC(x,y):
    '''
    Calculate the SRCC of the two 1D tensors
    Args:
        x,y are 1D tensors
    '''
    x = np.array(x)
    y = np.array(y)
    return spearmanr(x,y)[0]
    
def SRCC_Mean(x,y):
    '''
    Calculate SRCC of the row-mean of 2D tensor : x,y
    Args:
        x,y are 2D tensors with the shape: [testset_size, 10]
    '''
    x_mean = x.mean(dim=1)
    y_mean = y.mean(dim=1) 
    return SRCC(x_mean,y_mean)

def SRCC_Std(x,y):
    '''
    Calculate SRCC of the row-std of 2D tensor : x,y
    Args:
        x,y are 2D tensors with the shape: [testset_size, 10]
    '''
    x_std = x.std(dim=1)
    y_std = y.std(dim=1)
    return SRCC(x_std,y_std)