

##
import sys
import vq_amm
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import pickle
from metrics import _cossim
import config as cf
from torchinfo import summary
import json
import pprint
from mlp_simple import MLP

from sklearn.metrics import f1_score,recall_score,precision_score
from r import resnet_tiny
from pq_amm_cnn import PQ_AMM_CNN
from metrics import _cossim
from r_amm import ResNet_Tiny_Manual
from v import TMAP
from v_amm import ViT_Manual
from m import MLPMixer

##
import math

##

dim=16
depth=5

model = MLPMixer(in_channels=1, image_size=cf.image_size[0], patch_size=cf.patch_size[1], num_classes=cf.num_classes,
                 dim=dim, depth=depth, token_dim=dim, channel_dim=dim)



summary(model)
total_params = sum(p.numel() for p in model.parameters())
#N_SUBSPACE=[2]*5
#N_SUBSPACE=list(range(1,20))
#K_CLUSTER=list(range(1,20))
N_SUBSPACE=[2]*14
K_CLUSTER=[256]*14

N_SUBSPACE_C,K_CLUSTER_C=N_SUBSPACE[:],K_CLUSTER[:]
# total 14 tables
N_Train, N_Test = -1,-1 # -1 if using all data
#N_Train, N_Test = -1,-1


def load_data_n_model(model_save_path):
    tensor_dict_path = model_save_path + '.tensor_dict.pkl'
    # Load the dictionary using pickle
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
    train_data, train_target, test_data, test_target = \
        tensor_dict['train_data'], tensor_dict['train_target'], tensor_dict['test_data'], tensor_dict['test_target']

    # define and load model
    model.load_state_dict(torch.load(model_save_path))
    #all_params = list(model.named_parameters())
    model.eval()
    # load csv for threshold
    df_threshold = pd.read_csv(model_save_path+".val_res.csv", header=0, sep=" ")
    best_threshold = df_threshold.opt_th.values[0]
    return train_data, train_target, test_data, test_target, model.state_dict(), best_threshold


##
# load model and data

model_save_path = "../dataset/mixer_demo/654.roms/mixer_demo.pkl"
#model_save_path = "../dataset/vit_demo/410.bwaves/vit_demo.pkl"

train_data, train_target, test_data, test_target, model_state_dict, best_threshold = load_data_n_model(model_save_path)
train_data, train_target, test_data, test_target = train_data[:N_Train], train_target[:N_Train], test_data[:N_Test], test_target[:N_Test]


##
# check correctness of manual implementation
y_score_by_whole_train = model(train_data).detach().numpy()
y_score_by_whole_test = model(test_data).detach().numpy()

print(y_score_by_whole_train.shape)
print("done")
##
output = model(train_data)


##
#
#
# class MLP_Mixer_Manual():
#     def __init__(self, model, N_SUBSPACE, K_CLUSTER):
#         self.n_subspace = N_SUBSPACE
#         self.k_cluster = K_CLUSTER
#         self.patch_rearrange = model.to_patch_embedding[0]
#         self.mix_layers =
#
#     def get_param(self, model_layer, param_type="parameters"):
#         if param_type == "parameters":
#             return [param.detach().numpy() for param in model_layer.parameters()]
#         elif param_type == "buffer":  # extract BN mean and var, remove track_num in buffer
#             return [param.detach().numpy() for param in model_layer.buffers() if param.numel() > 1]
#         else:
#             raise ValueError("Invalid type in model layer to get parameters")
#
#
#     def forward(self,input_data, mm_type='exact'):
#
#         return layer_res, mm_res
#
#     def forward_exact(self,input_data):
#         return self.forward(input_data, mm_type='exact')
#
