#!/bin/bash
# 410, 433, 437, 462; 602, 605, 619, 621
APPS=("437.leslie3d-s0.txt.xz" "433.milc-s0.txt.xz") 

for app in ${APPS[@]}; do
    nohup python src/train.py $app vit 0 & 
    python src/train.py $app vitt 0
    python src/train_kd.py $app vitt vit 0 0.75
done

