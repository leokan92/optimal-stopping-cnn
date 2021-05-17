# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:26:26 2021

@author: leona
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:24:46 2021

@author: leona

LSTM class 
"""
from torch import nn

class Neural_Net_NN(nn.Module):
    def __init__(self, M,shape_input):
        super(Neural_Net_NN, self).__init__()
        self.dense1 = nn.Linear(shape_input,M)
        self.dense2 = nn.Linear(M,1)
        self.bachnorm1 = nn.BatchNorm1d(M,momentum = None,track_running_stats = False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.bachnorm1(self.relu(self.dense1(x)))
        x = self.relu(self.dense2(x))
        return x