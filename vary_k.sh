#!/bin/bash

APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")
K_VAL=(32 64 128 512 1024 2048)

for app1 in "${APPS[@]}"; do
    #python src/preprocess.py "$app1" 4
    #python src/train_kd.py "$app1" rt rs 0
    for k in "${K_VAL[@]}"; do 
        python src/2_resnet.py "$app1" rs "$k,$k,$k,$k,$k" "3,4,8,16,16" 2 
    done 
done

