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
from rdkit import Chem
from functools import partial
from multiprocessing import cpu_count, Pool
from copy import deepcopy
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
from functools import reduce
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning
from torch import norm

#Define siamese encoder
class enc_graph(LightningModule):
    def __init__(self,params):
        super(enc_graph, self).__init__()

        ### encode smiles
        #atoms0 = tf.keras.layers.InputLayer(name='atom_inputs', input_shape=(params["max_atoms"], params["num_atom_features"],),dtype = 'float32')
        #bonds = tf.keras.layers.InputLayer(name='bond_inputs', input_shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"],),dtype = 'float32')
        #edges = tf.keras.layers.InputLayer(name='edge_inputs', input_shape=(params["max_atoms"], params["max_degree"],), dtype='int32')

        self.g1 = NeuralGraphHidden(params["num_atom_features"],params["num_bond_features"],params["graph_conv_width"][0],params["max_degree"] , activ = None, bias = True)
        self.bn1 = nn.BatchNorm1d(num_features=params["graph_conv_width"][0],momentum=0.6)

        self.g2 = NeuralGraphHidden(params["graph_conv_width"][0],params["num_bond_features"],params["graph_conv_width"][1],params["max_degree"] , activ = None, bias = True)
        self.bn2 = nn.BatchNorm1d(num_features=params["graph_conv_width"][1],momentum=0.6)

        self.g3 = NeuralGraphHidden(params["graph_conv_width"][1],params["num_bond_features"],params["graph_conv_width"][2],params["max_degree"] , activ = None, bias = True)
        self.bn3 = nn.BatchNorm1d(num_features=params["graph_conv_width"][2],momentum=0.6)


        self.conv1d=nn.Conv1d(params["conv1d_in"], params["conv1d_out"], params["kernel_size"],bias=False)
        nn.init.xavier_normal_(self.conv1d.weight)
        self.bn4= nn.BatchNorm1d(num_features=int((params["graph_conv_width"][1]-params["kernel_size"])/params["kernel_size"]+1),momentum=0.6)
        self.dropout = nn.Dropout(params["dropout_encoder"])
        #self.dropout=params["dropout_encoder"]

    def forward(self,atoms,bonds,edges):
        x1 = self.g1([atoms,bonds,edges])
        x1=torch.transpose(x1,1,2)
        x1 = self.bn1(x1)
        x1=torch.transpose(x1,1,2)
        x1 = F.relu(x1)

        x2 = self.g2([x1,bonds,edges])
        x2=torch.transpose(x2,1,2)
        x2 = self.bn2(x2)
        x2=torch.transpose(x2,1,2)
        x2 = F.relu(x2)

        x3 = self.g3([x2,bonds,edges])
        x3=torch.transpose(x3,1,2)
        x3 = self.bn3(x3)
        x3=torch.transpose(x3,1,2)
        x3 = F.relu(x3)

        x4 = self.conv1d(x3)
        x4=torch.transpose(x4,1,2)
        x4 = self.bn4(x4)
        x4=torch.transpose(x4,1,2)
        x4 = F.relu(x4)
        x4 = self.dropout(x4)

        return x4


    #End of encoding
    #graph_encoder = keras.Model(inputs=[atoms0, bonds, edges], outputs= g4)

    #print(graph_encoder.summary())
    #return graph_encoder

#Define operations of distance module after the siamese encoders
class siamese_model(LightningModule):
    def __init__(self,params):
        super(siamese_model, self).__init__()

        self.learning_rate = params["lr"]
        self.l2reg= params["l2reg"]
        self.encoder = enc_graph(params)
        self.conv1 = nn.Conv1d(params["conv1d_dist_in"][0], params["conv1d_dist_out"][0], params["conv1d_dist_kernels"][0],bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=int((params["graph_conv_width"][1]-params["conv1d_dist_kernels"][0])/params["conv1d_dist_kernels"][0]+1),momentum=0.6)
        self.drop_dist1 = nn.Dropout(params["dropout_dist"])
        #self.drop_dist1 = params["dropout_dist"]
        self.conv2 = nn.Conv1d(params["conv1d_dist_in"][1], params["conv1d_dist_out"][1], params["conv1d_dist_kernels"][1],bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=int((params["graph_conv_width"][1]-params["conv1d_dist_kernels"][1])/params["conv1d_dist_kernels"][1]+1),momentum=0.6)
        self.drop_dist2 = nn.Dropout(params["dropout_dist"])
        #self.drop_dist2 = params["dropout_dist"]
        self.pool = nn.MaxPool1d(params["pool_size"])
        self.out_pool=int((params["conv1d_dist_out"][1]-params["pool_size"])/params["pool_size"]+1)
        self.bn3 = nn.BatchNorm1d(num_features=int((params["graph_conv_width"][1]-params["conv1d_dist_kernels"][1])/params["conv1d_dist_kernels"][1]+1),momentum=0.6)
        self.flatten = nn.Flatten()
        self.out_flat=int(self.out_pool*params["graph_conv_width"][2])
        self.dense1 = nn.Linear(self.out_flat, params["dense_size"][0], bias=True)
        self.bn4 =  nn.BatchNorm1d(num_features=params["dense_size"][0],momentum=0.6)
        self.drop_dist4 = nn.Dropout(params["dropout_dist"])
        #self.drop_dist4 = params["dropout_dist"]
        self.dense2 = nn.Linear(params["dense_size"][0], params["dense_size"][1], bias=True)
        self.bn5 = nn.BatchNorm1d(num_features=params["dense_size"][1],momentum=0.6)
        self.drop_dist5 = nn.Dropout(params["dropout_dist"])
        #self.drop_dist5 = params["dropout_dist"]
        self.dense3 = nn.Linear(params["dense_size"][1], params["dense_size"][2], bias=True)
        self.bn6 = nn.BatchNorm1d(num_features=params["dense_size"][2],momentum=0.6)
        self.drop_dist6 = nn.Dropout(params["dropout_dist"])
        #self.drop_dist6 = params["dropout_dist"]
        if params["ConGauss"]:
            self.gaussian = ConGaussianLayer(1,params["dense_size"][2])
        else:
            self.gaussian = GaussianLayer(1,params["dense_size"][2]) #default used most of the time

        #Initialize weigts
        nn.init.xavier_normal_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.conv2.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.dense1.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_normal_(self.dense2.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense2.bias)
        nn.init.xavier_normal_(self.dense3.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense3.bias)

    def forward(self,atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2):

        encoded_1 = self.encoder(atoms0_1,bonds_1,edges_1)
        encoded_2 = self.encoder(atoms0_2,bonds_2,edges_2)

        L1_distance=abs(encoded_1-encoded_2)

        x = self.conv1(L1_distance)
        x=torch.transpose(x,1,2)
        x = self.bn1(x)
        x=torch.transpose(x,1,2)
        x = F.relu(x)
        x = self.drop_dist1(x)

        x = self.conv2(x)
        x=torch.transpose(x,1,2)
        x = self.bn2(x)
        x=torch.transpose(x,1,2)
        x = F.relu(x)
        x = self.drop_dist2(x)

        x = torch.transpose(x,1,2)
        x = self.pool(x)
        #x=x.transpose(1,2)
        x = self.bn3(x)
        x=torch.transpose(x,1,2)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop_dist4(x)


        x = self.dense2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.drop_dist5(x)

        x = self.dense3(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.drop_dist6(x)

        #Final Gaussian Layer to predict mean distance and standard deaviation of distance
        mu, sigma = self.gaussian(x)
        out=torch.cat([mu,sigma],dim=-1)
        
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        atom1,bond1,edge1,atom2,bond2,edge2,y_true = batch
        y_pred = self(atom1,bond1,edge1,atom2,bond2,edge2)
        reg=norm(self.encoder.g1.inner_3D_layers[0].weight)**2 + norm(self.encoder.g2.inner_3D_layers[0].weight)**2
        for j in range(1,5):
            reg=reg+norm(self.encoder.g1.inner_3D_layers[j].weight)**2 + norm(self.encoder.g2.inner_3D_layers[j].weight)**2
        reg=reg + norm(self.dense1.weight)**2+norm(self.dense2.weight)**2+norm(self.dense3.weight)**2
        loss = custom_loss(y_true,y_pred) + self.l2reg*reg
        return(loss)
