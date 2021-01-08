from __future__ import division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import torch
import torch.nn as nn
from torch.functional import F
import os
import random
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import re
from functools import partial
from multiprocessing import cpu_count, Pool
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


#Define custom metrics for evaluation
def r_square(y_true, y_pred):
    #(y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    y_pred=y_pred[:,0]
    SS_res =  torch.sum(torch.square(y_true - y_pred))
    SS_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
    return (1 - SS_res/(SS_tot + 1e-7))

def get_cindex(y_true, y_pred):
    #(y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    y_pred=y_pred[:,0]
    g = torch.sub(y_pred.unsqueeze(-1), y_pred)
    g = (g == 0.0).type(torch.FloatTensor) * 0.5 + (g > 0.0).type(torch.FloatTensor)

    f = torch.sub(y_true.unsqueeze(-1), y_true) > 0.0
    f = torch.tril(f.type(torch.FloatTensor))

    g = torch.sum(g*f)
    f = torch.sum(f)

    return torch.where(g==0.0, torch.tensor(0.0), g/f)

def pearson_r(y_true, y_pred):
    #(y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    y_pred=y_pred[:,0]
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    x_square_sum = torch.sum(xm * xm)
    y_square_sum = torch.sum(ym * ym)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return torch.mean(r)

def custom_mse(y_true,y_pred):
    #(y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    y_pred=y_pred[:,0]
    er=torch.mean(torch.square(y_pred - y_true))
    return er

def mse_sliced(y_true,y_pred,th):
    def mse_similars(y_true,y_pred):
        #(y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
        y_pred=y_pred[:,0]
        condition = y_pred<=th
        indices = torch.where(condition)
        slice_true=y_true[indices[0]]
        slice_pred=y_pred[indices[0]]
        mse_sliced = torch.mean(torch.square(slice_pred - slice_true))
        return mse_sliced
    return mse_similars

#Model evaluation function
def model_evaluate(y_pred,Y_cold,thresh,df_cold):
    true = np.reshape(Y_cold,len(df_cold))
    pred = np.reshape(y_pred,len(df_cold))
    cor = np.corrcoef(true,pred)
    mse_all = sklearn.metrics.mean_squared_error(true,pred)
    # calculate mse of similars
    if (len(pred[np.where(pred<=thresh)])>0):
        mse_sims = sklearn.metrics.mean_squared_error(true[pred<=thresh],pred[pred<=thresh])
    else:
        mse_sims = "None"
    # turn to categorical to calculate precision and accuracy
    true_cat = true <= thresh
    pred_cat = pred <= thresh
    pos = np.sum(pred_cat)
    # calculate accuracy and fpr and precision
    tn, fp, fn, tp=confusion_matrix(true_cat,pred_cat).ravel() #see sklearn.metrics.confusion_matrix documentation
    if (len(pred[np.where(pred<=thresh)])>0):
        prec = tp/(fp+tp)
    else: 
        prec = "None"
    fpr=fp/(fp+tn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    result =pd.DataFrame({'cor' : cor[0,1], 'mse_all' : mse_all, 'mse_similars' : mse_sims,'precision': prec, 'accuracy': acc,
                         'FPR':fpr,'positives' : pos}, index=[0])
    return(result)
