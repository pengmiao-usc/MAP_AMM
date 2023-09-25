#!/bin/bash
# 410, 433, 437, 462; 602, 605, 619, 621
APPS=("410.bwaves-s0.txt.xz" "433.milc-s0.txt.xz" "437.leslie3d-s0.txt.xz" "462.libquantum-s0.txt.xz" "602.gcc-s0.txt.xz" "605.mcf-s0.txt.xz" "619.lbm-s0.txt.xz" "621.wrf-s0.txt.xz")

for app in ${APPS[@]}; do
    python src/train.py $app vit 1
done

