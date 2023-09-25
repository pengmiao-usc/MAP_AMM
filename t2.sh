#!/bin/bash
# 410, 433, 437, 462; 602, 605, 619, 621
APPS=("410.bwaves-s0.txt.xz" "462.libquantum-s0.txt.xz") 

for app in ${APPS[@]}; do
    nohup python src/train.py $app vit 1 &
    python src/train.py $app vitt 1 
    python src/train_kd.py $app vitt vit 1 0.75
done

