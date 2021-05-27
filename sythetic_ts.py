# -*- coding: utf-8 -*-
"""
Non-linear autocorrilated moving average time-series generation
"""

import timesynth as ts
import numpy as np
import random
from fbm import FBM
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing


#############################################################
#creating non-linear autocorrelated time-series with noise
#############################################################

def NARAM_sample(S0,K,N,d,batch_size,order):
    num_cores = multiprocessing.cpu_count()
    def NARAM_sample_in(S0,K,N,d,i,order):
        assets_eval = []
        for j in range(0,d):
            seed = random.randint(0,999999999)
            # Initializing TimeSampler
            time_sampler = ts.TimeSampler(stop_time=N-1)
            # Sampling irregular time samples
            times = time_sampler.sample_regular_time(resolution=1.)
            
            # Take Samples
            narma_signal = ts.signals.NARMA(order=order,seed = seed)
            series = ts.TimeSeries(narma_signal)
            samples, _, _ = series.sample(times)
            
            samples = samples-np.random.choice(samples)
        
            assets_eval.append(np.concatenate((np.array([S0]),S0+np.cumsum(S0*samples))))
        return assets_eval
    
    results = Parallel(n_jobs=num_cores)(delayed(NARAM_sample_in)(S0,K,N,d,i,order) for i in range(0,batch_size))
    return np.asarray(results)


def AR_sample(S0,K,N,d,batch_size):
    num_cores = multiprocessing.cpu_count()
    def AR_sample_in(S0,K,N,d,i):
        assets_eval = []
        for j in range(0,d):
            time_sampler = ts.TimeSampler(stop_time=N-1)
            times = time_sampler.sample_regular_time(resolution=1.)
            red_noise = ts.noise.RedNoise(std=0.6, tau=0.8)
            ar_p = ts.signals.AutoRegressive(ar_param=[1.5, -0.75])
            ar_p_series = ts.TimeSeries(signal_generator=ar_p, noise_generator=red_noise)
            samples, _, _ = ar_p_series.sample(times)
            samples_scaled = samples/np.max(samples)
            
            assets_eval.append(np.concatenate((np.array([S0]),S0+np.cumsum(S0*samples_scaled/10))))
        return assets_eval
    
    results = Parallel(n_jobs=num_cores)(delayed(AR_sample_in)(S0,K,N,d,i) for i in range(0,batch_size))
    return np.asarray(results)


def FBM_sample(S0,K,N,d,batch_size):
    num_cores = multiprocessing.cpu_count()
    f = FBM(n=N-1, hurst=.70, method='cholesky')
    def FBM_sample_in(S0,K,N,d,i):
        assets_eval = []
        for j in range(0,d):
            fbm_sample = f.fbm()
            assets_eval.append(S0*fbm_sample)
        return assets_eval
    results = Parallel(n_jobs=num_cores)(delayed(FBM_sample_in)(S0,K,N,d,i) for i in range(0,batch_size))
    return np.asarray(results)


def harmonic_sample(S0,K,N,d,batch_size):
    num_cores = multiprocessing.cpu_count()
    def harmonic_sample_in(S0,K,N,d,i):
        assets_eval = []
        for j in range(0,d):
            time_sampler = ts.TimeSampler(stop_time=10)
            irregular_time_samples = time_sampler.sample_irregular_time(num_points=N*2-1, keep_percentage=50)
            sinusoid = ts.signals.Sinusoidal(amplitude = 0.2,frequency=0.3)
            white_noise = ts.noise.GaussianNoise(std=0.15)
            timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
            samples, signals, errors = timeseries.sample(irregular_time_samples)
            sinusoid_2 = ts.signals.Sinusoidal(amplitude = 0.2,frequency=2)
            timeseries_2 = ts.TimeSeries(sinusoid_2, noise_generator=white_noise)
            samples_2, signals, errors = timeseries_2.sample(irregular_time_samples)
            samples_3 = samples+samples_2
            assets_eval.append(np.concatenate((np.array([S0]),S0+np.cumsum(S0*samples_3))))
        return assets_eval
    results = Parallel(n_jobs=num_cores)(delayed(harmonic_sample_in)(S0,K,N,d,i) for i in range(0,batch_size))
    return np.asarray(results)


def return_real_data_sample(path,file,N,d,batch_size,S0,sample_type):
    num_cores = multiprocessing.cpu_count()
    df = pd.read_csv(path+file,sep=';',thousands=',')
    #df = pd.read_csv(path+file,sep=',')
    returns = np.diff(df['Close']) / df['Close'][1:]
    if sample_type == 'none':
        if sample_type == 'train':
            returns = np.diff(df['Close']) / df['Close'][1+int(len(df['Close'])*0.3):]
        else:
            returns = np.diff(df['Close']) / df['Close'][1:int(len(df['Close'])*0.3)]   
    def rand_sample(S0,N,d,returns,i):
        asset_list = []
        for j in range(0,d):
            rand = np.random.randint(0,len(returns)-N-1)
            asset_list.append(np.concatenate((np.array([S0]),S0+np.cumsum(S0*returns[rand:rand+N-1].to_numpy()*10))))
        return asset_list
    results = Parallel(n_jobs=num_cores)(delayed(rand_sample)(S0,N,d,returns,i) for i in range(0,batch_size))
    return np.asarray(results)




