import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import os
import random
import sklearn
import re
#from NGF.utils import filter_func_args, mol_shapes_to_dims
#import NGF.utils
import NGF_layers.features
import NGF_layers.graph_layers
from NGF_layers.features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, bond_features, num_atom_features, num_bond_features, padaxis, tensorise_smiles #, concat_mol_tensors
from NGF_layers.graph_layers import NeuralGraphHidden
from math import ceil
from sklearn.metrics import mean_squared_error
from utility.gaussian import GaussianLayer, custom_loss, ConGaussianLayer
from utility.evaluator import r_square, get_cindex, pearson_r,custom_mse, mse_sliced, model_evaluate
from utility.Generator import train_generator,preds_generator
from deepSIBA_model import enc_graph, siamese_model
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
from torch import norm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import dill
from functools import partial
sns.set()


#model_params
model_params = {
    "max_atoms" : int(60), "num_atom_features" : int(62), "max_degree" : int(5), "num_bond_features" : int(6),
    "graph_conv_width" : [128,128,128], "conv1d_in" : int(60), "conv1d_out" : int(32), "kernel_size" : int(1), "dropout_encoder" : 0.25,
    "conv1d_dist_in" : [32,16], "conv1d_dist_out" : [16,16], "conv1d_dist_kernels" : [1,1], "dropout_dist" : 0.25, "pool_size" : int(4),
    "dense_size" : [256,128,128], "l2reg" : 0.01, "dist_thresh" : 0.2, "lr" : 0.001 ,"ConGauss": True
}


train_params = {
    "cell_line" : "a375", "split" : "train_test_split", "number_folds" : [0],
    "output_dir" : "results",
    "batch_size" : int(128), "epochs" : int(10), 
    "N_ensemble" : int(1), "nmodel_start" : int(0), "prec_threshold" : 0.2,
    "Pre_training" : False,
    "Pre_trained_cell_dir" : '',
    "pattern_to_load" : 'siam_no_augment_',
    "model_id_to_load" : "20",
    "test_value_norm" : True,
    "predict_batch_size":int(1024)
}


#Load data
get_all = []
if train_params["split"] == "train_test_split":
  outer_loop = train_params["number_folds"]
elif train_params["split"] == "5_fold_cv_split":
  outer_loop = train_params["number_folds"]
elif train_params["split"] == "alldata":
  outer_loop = train_params["number_folds"]
#Load unique smiles and tensorize them
smiles = pd.read_csv("data/" + train_params["cell_line"] + "/" + train_params["cell_line"] + "q1smiles.csv", index_col=0)
X_atoms, X_bonds, X_edges = tensorise_smiles(smiles.x, model_params["max_degree"], model_params["max_atoms"])
smiles=list(smiles['x'])

df = pd.read_csv("data/" + train_params["cell_line"] + "/" + "train_test_split/" + "train.csv",index_col=0).reset_index(drop=True)
df_cold = pd.read_csv("data/" + train_params["cell_line"] + "/" + "train_test_split/" + "test.csv",index_col=0).reset_index(drop=True)
smiles_cold = list(set(list(df_cold['rdkit.x'])+list(df_cold['rdkit.y'])))
X_atoms_cold, X_bonds_cold, X_edges_cold = tensorise_smiles(smiles_cold,  model_params["max_degree"], model_params["max_atoms"])
#X_atoms_cold=X_atoms_cold.astype('float64')
#X_bonds_cold=X_bonds_cold.astype('float64')
#X_edges_cold=X_edges_cold.astype('int64')
if train_params["test_value_norm"]:
  Y_cold = df_cold.value
else:
  Y_cold = df_cold.value
  Y_cold = Y_cold/2

i=0
Path(train_params["output_dir"] + "/" + "fold_%s/models"%i).mkdir(parents=True, exist_ok=True)
cold_preds_mus = []
cold_preds_sigmas = []
n = train_params["nmodel_start"]


def get_default_device():
  if torch.cuda.is_available():
    print('cuda mode')
    return torch.device('cuda')
  else:
    print('cpu mode')
  return torch.device('cpu')
device=get_default_device()

def to_device(data,device):
  if isinstance(data,(list,tuple)):
    return [to_device(x,device) for x in data]
  return data.to(device)

class train_generator(Dataset):

  def __init__(self, data,smiles,X_atoms, X_bonds, X_edges):
    self.df=data
    self.smiles=smiles
    self.X_atoms=X_atoms
    self.X_bonds=X_bonds
    self.X_edges=X_edges
    
  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    smi1=self.df['rdkit.x'][index]
    smi2=self.df['rdkit.y'][index]
    d=self.df.value[index]/2
    ind1=self.smiles.index(smi1)
    ind2=self.smiles.index(smi2)
    atom_1=torch.tensor(self.X_atoms[ind1])
    atom_1=torch.tensor(self.X_atoms[ind1])
    bond_1=torch.tensor(self.X_bonds[ind1])
    edge_1=torch.tensor(self.X_edges[ind1])
    atom_2=torch.tensor(self.X_atoms[ind2])
    bond_2=torch.tensor(self.X_bonds[ind2])
    edge_2=torch.tensor(self.X_edges[ind2])
    return atom_1,bond_1,edge_1,atom_2,bond_2,edge_2,torch.tensor(d)

class DeviceDataLoader():
  def __init__(self,dl,device):
    self.dl=dl
    self.device=device
  def __iter__(self):
    for b in self.dl:
      yield to_device(b,self.device)

  def __len__(self):
    """Number of batches"""
    return len(self.dl)

#Parameter space
# defining the space
fspace = {
    'batch_size' : hp.quniform('batch_size', 64,256,32),
    'lr' : hp.uniform('lr', 0.0005, 0.01),
    'l2reg' : hp.uniform('l2reg', 0.0005, 0.1)
}

def objective(fspace, train_params, model_params,df,smiles,X_atoms, X_bonds, X_edges):
    accs = []
    #model_params["lr"]= fspace["lr"]
    #model_params["l2reg"]= fspace["l2reg"]
    #train_params["batch_size"]= fspace["batch_size"]

    bs = int(fspace["batch_size"])
    NUM_EPOCHS = train_params["epochs"]

    #num_workers=12 mporei na mpei ki ayto sto DataLoader
    train_loader = DataLoader(trainGen,
                          batch_size=bs,
                          num_workers=6,
                          shuffle=True)
    for i in range(10):
        deepsiba = siamese_model(model_params)
        df = df.sample(frac=1).reset_index(drop=True)
        NUM_TRAIN = len(df)
        NUM_STEPS=ceil(NUM_TRAIN/bs)
        trainGen=train_generator(df,smiles,X_atoms, X_bonds, X_edges)

        #num_workers=12 mporei na mpei ki ayto sto DataLoader
        train_loader = DataLoader(trainGen,
                                  batch_size=bs,
                                  num_workers=6,
                                  shuffle=True)
        train_loader=DeviceDataLoader(train_loader,device)
        deepsiba=to_device(deepsiba,device)
        adam = torch.optim.Adam(deepsiba.parameters(),lr=fspace["lr"],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, 'min',factor=0.5,patience=3, min_lr=0.00001, eps=1e-5,verbose=True)

        for epoch in range(NUM_EPOCHS):
            deepsiba.train()
            for atom1,bond1,edge1,atom2,bond2,edge2,y_true in train_loader:
                y_pred = deepsiba(atom1,bond1,edge1,atom2,bond2,edge2)
                reg=norm(deepsiba.encoder.g1.inner_3D_layers[0].weight)**2 + norm(deepsiba.encoder.g2.inner_3D_layers[0].weight)**2
                for j in range(1,model_params["max_degree"]):
                    reg=reg+norm(deepsiba.encoder.g1.inner_3D_layers[j].weight)**2 + norm(deepsiba.encoder.g2.inner_3D_layers[j].weight)**2
                reg=reg + norm(deepsiba.dense1.weight)**2+norm(deepsiba.dense2.weight)**2+norm(deepsiba.dense3.weight)**2
                loss = custom_loss(y_true,y_pred) + fspace["l2reg"]*reg
                r=r_square(y_true,y_pred)
                pear=pearson_r(y_true,y_pred)
                cindex=get_cindex(y_true,y_pred)
                loss.backward()
                adam.step()
        
        accs.append(float(r.cpu()))
    ave_acc = np.mean(accs,axis = 0)
    return {'loss': -ave_acc ,  'status': STATUS_OK}

fmin_objective = partial(objective, train_params, model_params,df,smiles,X_atoms, X_bonds, X_edges)

def run_trials():

    trials_step = 10  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("torch_hyperparam.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn = fmin_objective, space = fspace, algo=tpe.suggest, max_evals=max_trials, trials=trials)

    print("Best:", best)
    
    # save the trials object
    with open("torch_hyperparam.hyperopt", "wb") as f:
        pickle.dump(trials, f)
    return(trials)

trials = run_trials()
