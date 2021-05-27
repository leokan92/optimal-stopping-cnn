# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:42:09 2021

@author: leona
"""

import pandas as pd
df = pd.read_excel('Datasets/Crude oil future contract.xlsx')
df.rename(columns={'Cushing, OK Crude Oil Future Contract 1 (Dollars per Barrel)':'Close'},inplace=True)
df = df.dropna(0)
df.to_csv('Datasets/crudeoil.csv',sep=';')
df[:int(len(df)*0.8)].to_csv('Datasets/crudeoil_train.csv',sep=';')
df[int(len(df)*0.8):].to_csv('Datasets/crudeoil_test.csv',sep=';')