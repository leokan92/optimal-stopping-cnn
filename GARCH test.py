# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:52:06 2021

@author: leona
"""

import pyflux

import numpy as np
import pyflux as pf
import pandas as pd
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt


jpm = DataReader('JPM',  'yahoo', datetime(2006,1,1), datetime(2016,3,10))
returns = pd.DataFrame(np.diff(np.log(jpm['Adj Close'].values)))
returns.index = jpm.index.values[1:jpm.index.values.shape[0]]
returns.columns = ['JPM Returns']

plt.figure(figsize=(15,5));
plt.plot(returns.index,returns);
plt.ylabel('Returns');
plt.title('JPM Returns');

plt.figure(figsize=(15,5))
plt.plot(returns.index, np.abs(returns))
plt.ylabel('Absolute Returns')
plt.title('JP Morgan Absolute Returns');

model = pf.GARCH(returns,p=1,q=1)
x = model.fit(method='BBVI', iterations=10000, optimizer='ADAM')
x.summary()


model.plot_predict_is(h=50,figsize=(15,5))

model.sample(10)