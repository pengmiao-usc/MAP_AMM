import os
import sys

import yaml
import vq_amm
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import pickle
from metrics import _cossim
from torchinfo import summary
import json
import pprint
import math
from sklearn.metrics import f1_score,recall_score,precision_score

from utils import select_model, replace_directory
from pq_amm_cnn import PQ_AMM_CNN
from metrics import _cossim
from r_amm import ResNet_Tiny_Manual

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

def evaluate_by_score(y_score, threshold, y_label):
    y_pred_bin = (y_score - np.array(threshold) > 0) * 1
    p, r, f1 = evaluate(y_label, y_pred_bin)
    return [p, r, f1]


def lut_info_summary(est_list):
    lut_shape_ls=[]
    lut_n = len(est_list)
    for est in est_list:
        lut = est.luts
        lut_shape_ls.append(lut.shape)
    lut_total_sz = sum(math.prod(value) for value in lut_shape_ls)
    return lut_n, lut_shape_ls, lut_total_sz

def layer_cossim(layer_exact, layer_amm):
    res = []
    n = len(layer_exact)
    for i in range(n):
        res.append(_cossim(layer_exact[i], layer_amm[i]))
    return res

def load_data_n_model(model_save_path, res_path):
    tensor_dict_path = model_save_path + '.tensor_dict.pkl'
    # Load the dictionary using pickle
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
    train_data, train_target, test_data, test_target = \
        tensor_dict['train_data'], tensor_dict['train_target'], tensor_dict['test_data'], tensor_dict['test_target']

    # define and load model
    model.load_state_dict(torch.load(model_save_path))
    all_params = list(model.named_parameters())

    # load json in res dir. for threshold
    with open(res_path+".val_res.json", "r") as json_file:
        data = json.load(json_file)
    
    validation_list = data.get("validation")
    best_threshold = validation_list[0].get("threshold")
#    highest_f1 = -1  
#    best_threshold = None

#    for entry in validation_list:
#        f1 = entry.get("f1")
#        threshold = entry.get("threshold")
#        
#        if f1 is not None and f1 > highest_f1:
#            highest_f1 = f1
#            best_threshold = threshold
    return train_data, train_target, test_data, test_target, all_params, best_threshold

##

##################################################################################################
# main
##################################################################################################


with open("params.yaml", "r") as p:
    params = yaml.safe_load(p)

model_dir = params["system"]["model"]
res_dir = params["system"]["res"]

app = sys.argv[1]
option = sys.argv[2]
gpu_id = sys.argv[3]

K_CLUSTER = [int(c) for c in sys.argv[4].split(",")]
N_SUBSPACE = [int(d) for d in sys.argv[5].split(",")]

model = select_model(option.split(".")[0])
model_save_path = os.path.join(model_dir, f"{app[:-7]}.{option}.pkl") 
res_path = replace_directory(model_save_path, res_dir) 

train_data, train_target, test_data, test_target, all_params, best_threshold = load_data_n_model(model_save_path, res_path)

res_path += ".k."+".".join(map(str, K_CLUSTER))+".n."+".".join(map(str, N_SUBSPACE))

##
# check correctness of manual implementation
y_score_by_whole_train = model(train_data).detach().numpy()
y_score_by_whole_test = model(test_data).detach().numpy()

resnet_manual_amm = ResNet_Tiny_Manual(model,N_SUBSPACE,K_CLUSTER)
layer_exact_res_train, mm_exact_res_train = resnet_manual_amm.forward_exact_bn_fold(train_data)
layer_exact_res_test, mm_exact_res_test  = resnet_manual_amm.forward_exact_bn_fold(test_data)
print("Manual and Torch results are equal (Train):", np.allclose(y_score_by_whole_train, layer_exact_res_train[-1], atol=1e-5))
print("Manual and Torch results are equal (Test):", np.allclose(y_score_by_whole_test, layer_exact_res_test[-1], atol=1e-5))

##
# train amm and eval amm

layer_amm_res_train, mm_amm_res_train = resnet_manual_amm.train_amm(train_data)
print("Cosine similarity between AMM and exact (Train):", _cossim(y_score_by_whole_train, layer_amm_res_train[-1]))

layer_amm_res_test, mm_amm_res_test  = resnet_manual_amm.eval_amm(test_data)
print("Cosine similarity between AMM and exact (Test):", _cossim(y_score_by_whole_test, layer_amm_res_test[-1]))

cossim_layer_train = layer_cossim(layer_exact_res_train, layer_amm_res_train)
cossim_layer_test = layer_cossim(layer_exact_res_test, layer_amm_res_test)

cossim_mm_train = layer_cossim(mm_exact_res_train, mm_amm_res_train)
cossim_mm_test = layer_cossim(mm_exact_res_test, mm_amm_res_test)

f1_exact_ts = evaluate_by_score(y_score_by_whole_test, best_threshold, test_target)
f1_est_ts = evaluate_by_score(layer_amm_res_test[-1], best_threshold, test_target)

total_params = sum(p.numel() for p in model.parameters())

##
# output report
lut_num, lut_shape_list, lut_total_size = lut_info_summary(resnet_manual_amm.amm_estimators)
report = {
    'model': {
        'name': 'ResNet-tiny',
        'layer': 8,
        'dim': 4,
        'f1': f1_exact_ts,
        'num_param': total_params
    },
    'estimator': {
        'method': 'PQ_KMEANS',
        'N_SUBSPACE': N_SUBSPACE,
        'K_CLUSTER': K_CLUSTER,
        'cossim_layer_train': cossim_layer_train,
        'cossim_layer_test': cossim_layer_test,
        'cossim_amm_train': cossim_mm_train,
        'cossim_amm_test': cossim_mm_test,
        'f1': f1_est_ts,
        'lut_num': lut_num,
        'lut_shapes': lut_shape_list,
        'lut_total_size': lut_total_size
    }
}

pprint.pprint(report, sort_dicts=False)
with open(res_path+'.amm_report.json', 'w') as json_file:
    json.dump(report, json_file, indent=2)


