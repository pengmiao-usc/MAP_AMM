#!/bin/bash

APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")

for app1 in "${APPS[@]}"; do
    python src/preprocess.py $app1 1
    python src/train.py $app1 ms 1 
    python src/1_mm.py $app1 ms 1
    python src/train.py $app1 mt 1
    python src/2_mm.py $app1 mt 1
    python src/train_kd.py $app1 mt ms 1
    python src/1_mm.py $app1 ms.stu 1
done
