#!/bin/bash

APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")

N_VALUES=(1 2 4 8)

for app1 in "${APPS[@]}"; do
    #python src/preprocess.py $app1 4 
    python src/train.py $app1 mt 4
    python src/train.py $app1 ms 4
    python src/train_kd.py $app1 mt ms 4 
    for n in "${N_VALUES[@]}"; do 
        python src/1_mm.py $app1 ms 4
        python src/1_mm.py $app1 ms.stu 4 
    done
    python src/generate.py $app1 ms 4
done
