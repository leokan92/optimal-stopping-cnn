# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:13:39 2021

@author: leona

Create Fractional Brownian motion asset prices
"""

from fbm import FBM
import numpy as np




def FBM_sample(S0,K,N,d,batch_size):
    f = FBM(n=N-1, hurst=0.75, method='cholesky')
    batch_sample = []
    for i in range(0,batch_size):
        assets_eval = []
        for j in range(0,d):
            fbm_sample = f.fbm()
            assets_eval.append(S0*fbm_sample)
        batch_sample.append(assets_eval)
    return np.asarray(batch_sample)


FBM_sample(100,100,50,2,8192)