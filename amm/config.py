import torch

# %%HW Configuration; fixed
BLOCK_BITS = 6
PAGE_BITS = 12  # 12
TOTAL_BITS = 64
BLOCK_NUM_BITS = TOTAL_BITS - BLOCK_BITS

# %% Input and labeling;
# %%%tunable
SPLIT_BITS = 6
LOOK_BACK = 9

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
Degree = 16
FILTER_SIZE = 16

dim = 64
channels = 1
