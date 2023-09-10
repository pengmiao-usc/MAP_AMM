import torch

# %%HW Configuration; fixed
BLOCK_BITS = 6
PAGE_BITS = 12  # 12
# SUPER_PAGE_BITS=13
TOTAL_BITS = 64
BLOCK_NUM_BITS = TOTAL_BITS - BLOCK_BITS

# %% Input and labeling;
# %%%tunable
SPLIT_BITS = 6
LOOK_BACK = 9
# LOOK_FORWARD=200#look forward for collect training labels
PRED_FORWARD = 128  # pred forward
DELTA_BOUND = 128
'''
bitmap:e.g. DELTA_BOUND=4; bitmap length = 2*DELTA_BOUND=8
    index: [0,1,2,3, 4, 5, 6, 7]
    value: [1,2,3,4,-4,-3,-2,-1]
value = index+1  ; <DELTA_BOUND
      = index - DELTA_BOUND ; >DELTA_BOUND
'''

# %%% fixed
BITMAP_SIZE = 2 * DELTA_BOUND
image_size = (LOOK_BACK + 1, BLOCK_NUM_BITS // SPLIT_BITS + 1)  # h,w
patch_size = (1, image_size[1])
num_classes = 2 * DELTA_BOUND

# %% filter
# Degree=2
# FILTER_SIZE=10
Degree = 16
FILTER_SIZE = 16

# %%
# Model; tunable
# %%%shape
'''
dim=8
depth=1
heads=2
mlp_dim=8
channels=1
context_gamma=0.2
'''
dim = 16
depth = 2
heads = 2
mlp_dim = 16
channels = 1
context_gamma = 0.2
# %% Model Definition

# %%% training
batch_size = 256
epochs = 50
# 200
lr = 2e-4
early_stop = 5
# %%% scheduler
gamma = 0.1
step_size = 20

# mlp_mixer_dim = 20
# mlp_mixer_depth = 2
# channel_dim = 32

mlp_mixer_dim = 16
mlp_mixer_depth = 3
channel_dim = 32

gpu_id = '0'
# device_id = [0, 1]
device_id = [0]

