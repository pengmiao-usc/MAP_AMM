#!/bin/bash
# 410, 433, 437, 462; 602, 605, 619, 621
APPS=("605.mcf-s0.txt.xz" "621.wrf-s2.txt.xz")

for app in ${APPS[@]}; do
    nohup python src/train.py $app vit 3 &
    python src/train.py $app vitt 3 
    python src/train_kd.py $app vitt vit 3 0.75
done

