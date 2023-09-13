##
import sys
import vq_amm
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import pickle
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
import time
##
import math

##
model = TMAP(
    image_size=cf.image_size,
    patch_size=cf.patch_size,
    num_classes=cf.num_classes,
    dim=cf.dim,
    depth=cf.depth,
    heads=cf.heads,
    mlp_dim=cf.mlp_dim,
    channels=cf.channels,
    dim_head=cf.mlp_dim
)

summary(model)
total_params = sum(p.numel() for p in model.parameters())
#N_SUBSPACE=[2]*5
#N_SUBSPACE=list(range(1,20))
#K_CLUSTER=list(range(1,20))
N_SUBSPACE=[2]*14
K_CLUSTER=[64]*14

N_SUBSPACE_C,K_CLUSTER_C=N_SUBSPACE[:],K_CLUSTER[:]
# total 14 tables
N_Train, N_Test = -1,-1 # -1 if using all data
#N_Train, N_Test = 10000,10000
##
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
    return [float(x) for x in res]

def load_data_n_model(model_save_path):
    tensor_dict_path = model_save_path + '.tensor_dict.pkl'
    # Load the dictionary using pickle
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
    train_data, train_target, test_data, test_target = \
        tensor_dict['train_data'], tensor_dict['train_target'], tensor_dict['test_data'], tensor_dict['test_target']

    # define and load model
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    #all_params = list(model.named_parameters())
    model.eval()
    # load csv for threshold
    df_threshold = pd.read_csv(model_save_path+".val_res.csv", header=0, sep=" ")
    best_threshold = df_threshold.opt_th.values[0]
    return train_data, train_target, test_data, test_target, model.state_dict(), best_threshold


##
# load model and data

#model_save_path = "../dataset/vit_demo/654.roms/vit_demo.pkl"
model_save_path = "../dataset/vit_demo/410.bwaves/vit_demo.pkl"

train_data, train_target, test_data, test_target, model_state_dict, best_threshold = load_data_n_model(model_save_path)
train_data, train_target, test_data, test_target = train_data[:N_Train], train_target[:N_Train], test_data[:N_Test], test_target[:N_Test]


##
# check correctness of manual implementation
y_score_by_whole_train = model(train_data).detach().numpy()
y_score_by_whole_test = model(test_data).detach().numpy()


vit_manual_amm = ViT_Manual(model, N_SUBSPACE, K_CLUSTER)

layer_exact_res_train, mm_exact_res_train = vit_manual_amm.forward_exact(train_data)
#print("Manual and Torch results are equal (Train):", np.allclose(y_score_by_whole_train, layer_exact_res_train[-1], atol=1e-5))
print("Manual and Torch results cosine similarity (Train):", _cossim(y_score_by_whole_train, layer_exact_res_train[-1]))

layer_exact_res_test, mm_exact_res_test = vit_manual_amm.forward_exact(test_data)
#print("Manual and Torch results are equal (Test):", np.allclose(y_score_by_whole_test, layer_exact_res_test[-1], atol=1e-5))
print("Manual and Torch results cosine similarity (Test):", _cossim(y_score_by_whole_test, layer_exact_res_test[-1]))

##
# UPDATE HERE FOR FINE-TUNING!
print("start table training with fine_tuning...")
layer_amm_res_train, mm_amm_res_train = vit_manual_amm.fine_tune(train_data,mm_exact_res_train.copy())
print("start table evaluation...")

start_time = time.time()
layer_amm_res_test, mm_amm_res_test  = vit_manual_amm.eval_amm(test_data)
print(f"Elapsed time: { time.time() - start_time} seconds")
##
print("Cosine similarity between AMM and exact (Train):", _cossim(y_score_by_whole_train, layer_amm_res_train[-1]))
print("Cosine similarity between AMM and exact (Test):", _cossim(y_score_by_whole_test, layer_amm_res_test[-1]))


##

cossim_layer_train = layer_cossim(layer_exact_res_train, layer_amm_res_train)
##
cossim_layer_test = layer_cossim(layer_exact_res_test, layer_amm_res_test)

cossim_mm_train = layer_cossim(mm_exact_res_train, mm_amm_res_train)
cossim_mm_test = layer_cossim(mm_exact_res_test, mm_amm_res_test)

f1_exact_ts = evaluate_by_score(y_score_by_whole_test, best_threshold, test_target)
f1_est_ts = evaluate_by_score(layer_amm_res_test[-1], best_threshold, test_target)

print("done")
##


# output report
lut_num, lut_shape_list, lut_total_size = lut_info_summary(vit_manual_amm.amm_estimators)
report = {
    'model': {
        'name': 'ViT',
        'layer': len(cossim_layer_train),
        'dim': cf.dim,
        'f1': f1_exact_ts,
        'num_param': total_params
    },
    'estimator': {
        'method': 'PQ_KMEANS',
        'N_SUBSPACE': N_SUBSPACE_C,
        'K_CLUSTER': K_CLUSTER_C,
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

pprint.pprint(report,sort_dicts=False)
with open(model_save_path+'.estimator_report_fine_tune64.json', 'w') as json_file:
    json.dump(report, json_file,indent=2)

