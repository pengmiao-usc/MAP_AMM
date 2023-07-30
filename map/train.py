import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'2 in tarim'
import warnings
warnings.filterwarnings('ignore')
import sys
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import pandas as pd
import config as cf
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
#from model import resnet18,resnet14,resnet50,resnet101,resnet152,resnet_tiny
#from preprocessing import read_load_trace_data,preprocessing_patch
from torch.autograd import Variable
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
from tqdm import tqdm
from data_loader import data_generator
import os
import pdb
from validation import run_val
import pickle
from torchinfo import summary

torch.manual_seed(100)

device=cf.device
batch_size=cf.batch_size
epochs = cf.epochs
lr = cf.lr
gamma = cf.gamma
step_size=cf.step_size
pred_num=cf.PRED_FORWARD
early_stop = cf.early_stop

#%%

# update: tab

from mlp_simple import MLP

model = MLP(input_size = torch.prod(torch.tensor(cf.image_size)*cf.channels),
            hidden_size = cf.dim,
            num_classes = cf.num_classes
            )
print(summary(model))
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
log=cf.Logger()

#%%

def train(ep,train_loader,model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target)in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        optimizer.zero_grad()
        output = model(data)
        #loss = F.binary_cross_entropy_with_logits(output, target)
        loss = F.binary_cross_entropy(output, target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, loading, model_save_path,train_loader,test_loader,lr):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        log.logger.info("-------------Model Loaded------------")
        
    best_loss=0
    early_stop=cf.early_stop
    model.to(device)
    for epoch in range(epochs):
        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        log.logger.info((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            log.logger.info("-------- Save Best Model! --------")
            early_stop=cf.early_stop
        else:
            early_stop-=1
            log.logger.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            log.logger.info("-------- Early Stop! --------")
            break
        #test(test_loader)
        #scheduler.step()

#%%
##########################################################################################################
# update: tab

def save_data_for_amm(model_save_path):
    
    all_train_data,all_train_targets, all_test_data, all_test_targets = [],[],[],[]
    
    print("saving tensor pickle data")
    for batch_idx, (data, target) in enumerate(train_loader):
        all_train_data.append(data)
        all_train_targets.append(target)
        
    for batch_idx, (data, target) in enumerate(test_loader):
        all_test_data.append(data)
        all_test_targets.append(target)
    
    
    all_train_data = torch.cat(all_train_data, dim=0).cpu()
    all_train_targets = torch.cat(all_train_targets, dim=0).cpu()
    all_test_data = torch.cat(all_test_data, dim=0).cpu()
    all_test_targets = torch.cat(all_test_targets, dim=0).cpu()
    tensor_dict = {"train_data": all_train_data,
                   "train_target": all_train_targets,
                   "test_data": all_test_data, 
                   "test_target": all_test_targets}
    
    with open(model_save_path+'.tensor_dict.pkl', 'wb') as f:
        pickle.dump(tensor_dict, f)
    
    print("tensor data saved")
    
    print("saving test dataframe")
    #train_df.to_pickle(res_root+"train_df.pkl")
    test_df[['id', 'cycle', 'addr', 'ip','block_address','future', 'y_score']].to_pickle(
        model_save_path+".test_df.pkl")
    
    print("done data saving for amm")
        
    return

#%%
##########################################################################################################


#file_path="/home/pengmiao/Disk/work/data/ML-DPC-S0/LoadTraces/654.roms-s0.txt.xz"
file_path="/data/pengmiao/ML-DPC-S0/LoadTraces/410.bwaves-s0.txt.xz"


res_root = "../dataset/mlp_demo/410.bwaves/"
#res_root = "../../dataset/mlp_demo/654.roms/"

if not os.path.exists(res_root):
    os.makedirs(res_root)

model_save_path=res_root+"mlp_demo.pkl"

log_path=model_save_path+".log"
SKIP_NUM=0
TRAIN_NUM = 1
TOTAL_NUM=2

loading=False
log_path=model_save_path+".log"
log.set_logger(log_path)
log.logger.info("%s"%file_path)
train_loader, test_loader, train_df, test_df = data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM)

log.logger.info("-------------Data Proccessed------------")
run_epoch(epochs, loading,model_save_path,train_loader,test_loader,lr=cf.lr)
run_val(test_loader,test_df,file_path,model_save_path)

# update: tab
save_data_for_amm(model_save_path)

#%%




#%%
'''
if __name__ == "__main__":

    file_path=sys.argv[1]
    model_save_path=sys.argv[2]
    TRAIN_NUM = int(sys.argv[3])
    TOTAL_NUM = int(sys.argv[4])
    SKIP_NUM = int(sys.argv[5])

    if os.path.isfile(model_save_path) :
       loading=True
    else:
       loading=False
    loading=False
    log_path=model_save_path+".log"
    log.set_logger(log_path)
    log.logger.info("%s"%file_path)
    train_loader, test_loader, test_df = data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    log.logger.info("-------------Data Proccessed------------")
    run_epoch(epochs, loading,model_save_path,train_loader,test_loader,lr=cf.lr)
    run_val(test_loader,test_df,file_path,model_save_path)
'''
