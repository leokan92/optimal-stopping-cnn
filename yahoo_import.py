#Importing datasets from Yahoo finance

import pandas_datareader as pdr
from datetime import datetime

symbols_list = ['BCE','VZ','T','IPG','DISH','ABEV','LEN','F','BTI','GPS','BP','COP','CVX','COG','LNG','ACNB','ORI','RE','AXP','C','DD','BHP','SCCO','OLN','MOS','INTC','HPQ','AMD','LOGI','ARW']
for symbol in symbols_list:
  df = pdr.get_data_yahoo(symbols= symbol, start=datetime(2000, 1, 1), end=datetime(2021, 6, 17))
  df_train = df[:int(len(df)*0.7)]
  df_train.to_csv(r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets\\'+symbol+'_train.csv')
  df_test = df[int(len(df)*0.7):]
  df_test.to_csv(r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets\\' +symbol+'_test.csv')  