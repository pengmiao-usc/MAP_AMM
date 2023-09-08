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

# %%% fixed
BITMAP_SIZE = 2 * DELTA_BOUND
image_size = (LOOK_BACK + 1, BLOCK_NUM_BITS // SPLIT_BITS + 1)  # h,w
patch_size = (1, image_size[1])
num_classes = 2 * DELTA_BOUND

dim=16
depth=2
heads=2
mlp_dim=16
channels=1
context_gamma=0.2

