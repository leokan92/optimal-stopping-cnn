# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:15:21 2021

@author: leona
"""

from torch import nn,cat

class Neural_Net_CNN(nn.Module):
    def __init__(self, M,channels,shape_input,mlp_input,batch_size):
        super(Neural_Net_CNN, self).__init__()
        self.last_n_channels = 3
        self.conv1 = nn.Conv1d(channels,self.last_n_channels*2,kernel_size  = 3)
        self.conv2 = nn.Conv1d(self.last_n_channels*2,self.last_n_channels,kernel_size  = 3)
        self.maxpool1 = nn.MaxPool1d(2,stride=2)
        self.maxpool2 = nn.MaxPool1d(2,stride=2)
        self.dense1 = nn.Linear((shape_input-4)*self.last_n_channels+mlp_input,M)
        #self.dense1 = nn.Linear(channels*shape_input+mlp_input,M)
        self.dense2 = nn.Linear(M,1)
        self.bachnorm1 = nn.BatchNorm1d(M,momentum = None,track_running_stats = False)
        self.bachnorm_conv1 = nn.BatchNorm1d(self.last_n_channels*2,momentum = None,track_running_stats = False)
        self.bachnorm_conv2 = nn.BatchNorm1d(self.last_n_channels,momentum = None,track_running_stats = False)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten(1,-1)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x1,x2):
        x1 = self.bachnorm_conv1(self.relu(self.conv1(x1)))
        x1 = self.bachnorm_conv2(self.relu(self.conv2(x1)))
        x1 = self.flat(x1)
        x = cat((x1,x2),1)
        x = self.drop(self.bachnorm1(self.relu(self.dense1(x))))
        x = self.relu(self.dense2(x))
        return x