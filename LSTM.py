# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:24:46 2021

@author: leona

LSTM class 
"""
import torch
from torch import nn

class Neural_Net_LSTM(nn.Module):
    def __init__(self, M,shape_input,batch_size):
        super(Neural_Net_LSTM, self).__init__()
        self.lstm = nn.LSTM(shape_input,M,batch_first=True,num_layers=1)
        #self.dense1 = nn.Linear(shape_input,M)
        self.dense1 = nn.Linear(M,M) #Used with the LSTM
        torch.nn.init.xavier_uniform_(self.dense1.weight)
        self.dense2 = nn.Linear(M,M)
        torch.nn.init.xavier_uniform_(self.dense2.weight)
        self.dense3 = nn.Linear(M,1)
        torch.nn.init.xavier_uniform_(self.dense3.weight)
        self.drop   = nn.Dropout(0.5)
        self.bachnorm_hidden = nn.BatchNorm1d(M,momentum = None,track_running_stats = False)
        self.bachnorm_input = nn.BatchNorm1d(shape_input,momentum = 0.65,track_running_stats = False)
        self.bachnorm_output = nn.BatchNorm1d(1,momentum = None,track_running_stats = False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.shape_input = shape_input
        self.hidden_cell = (torch.zeros(1,batch_size,M),torch.zeros(1,batch_size,M))
        
    def forward(self, x):
        x = self.bachnorm_input(x)
        lstm_out, self.hidden_cell = self.lstm(torch.unsqueeze(x,1), self.hidden_cell)
        x = self.bachnorm_hidden(self.relu(self.dense1(torch.squeeze(lstm_out,1))))
        x = self.drop(self.bachnorm_hidden(self.relu(self.dense2(x))))
        x = self.relu(self.dense3(x))
        return x