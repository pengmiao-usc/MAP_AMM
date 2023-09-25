#!/bin/bash
# 410, 433, 437, 462; 602, 605, 619, 621
APPS=("602.gcc-s0.txt.xz" "619.lbm-s0.txt.xz") 

for app in ${APPS[@]}; do
    nohup python src/train.py $app vit 2 &
    python src/train.py $app vitt 2 
    python src/train_kd.py $app vitt vit 2 0.75
done

