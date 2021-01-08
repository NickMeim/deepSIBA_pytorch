from __future__ import absolute_import,division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import torch
import torch.nn as nn
from torch.functional import F
from tqdm.auto import tqdm
import os
import random
import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
import re
from tempfile import TemporaryFile
from functools import partial
from multiprocessing import cpu_count, Pool
from copy import deepcopy
from math import ceil

#Define custom loss and gaussian layer
def custom_loss(y_true, y_pred):
    #(y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    sigma = y_pred[:,1]
    y_pred = y_pred[:,0]
    return torch.mean(0.5*torch.log(sigma) + 0.5*torch.div(torch.square(y_true - y_pred), sigma)) + 1e-6

class GaussianLayer(nn.Module):
    def __init__(self, output_dim,input_shape, **kwargs):
        self.output_dim = output_dim
        self.input_shape=input_shape
        super(GaussianLayer, self).__init__(**kwargs)
        
        self.linear1=nn.Linear(input_shape, output_dim, bias=True)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        self.linear2=nn.Linear(input_shape, output_dim, bias=True)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x):
        output_mu  = self.linear1(x)
        output_sig = self.linear2(x)
        output_sig_pos = torch.log(1 + torch.exp(output_sig)) + 1e-06  
        return [output_mu, output_sig_pos]

class ConGaussianLayer(nn.Module):
    def __init__(self, output_dim,input_shape, **kwargs):
        self.output_dim = output_dim
        self.input_shape=input_shape
        super(ConGaussianLayer, self).__init__(**kwargs)
        
        self.linear1=nn.Linear(input_shape, output_dim, bias=True)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        self.linear2=nn.Linear(input_shape, output_dim, bias=True)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x):
        output_mu  = self.linear1(x)
        output_mu  = F.relu(output_mu)
        output_mu[output_mu > 1] = 1
        output_sig = self.linear2(x)
        output_sig_pos = torch.log(1 + torch.exp(output_sig)) + 1e-06  
        output_sig_pos  = F.relu(output_sig_pos) + 1e-06
        output_sig_pos[output_sig_pos > 1] = 1
        return [output_mu, output_sig_pos]
