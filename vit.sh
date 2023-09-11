#!/bin/bash

GPU_NO=2

OPTION="rs"
AMM_FILE="2_resnet.py"

APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")

for APP in "${APPS[@]}"; do
  # Train NN
  python src/train.py "$APP" $OPTION $GPU_NO
  
  python src/generate.py $APP $OPTION $GPU_NO

  for k in 16 32 64 128 256 512 1024 2048; do
    K_VALUES=$(printf "$k,"%.0s {1..4})$k
    N_VALUES=$(printf "2,"%.0s {1..4})2

    python src/$AMM_FILE $APP $OPTION $K_VALUES $N_VALUES $GPU_NO
    python src/generate_amm.py $APP $OPTION $K_VALUES $N_VALUES $GPU_NO  
  done

  for n in 1 2 4 8; do
    N_VALUES=$(printf "$n,"%.0s {1..4})$n
    K_VALUES=$(printf "256,"%.0s {1..4})256

    python src/$AMM_FILE $APP $OPTION $K_VALUES $N_VALUES $GPU_NO
    python src/generate_amm.py $APP $OPTION $K_VALUES $N_VALUES $GPU_NO 
  done

done

echo "All tasks completed!"