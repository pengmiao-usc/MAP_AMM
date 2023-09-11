#!/bin/bash
APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz") 
ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) 
for app1 in "${APPS[@]}"; do 
    for a in "${ALPHAS[@]}"; do
        python src/train_kd.py $app1 rt rs 3 $a 5 
    done
done
