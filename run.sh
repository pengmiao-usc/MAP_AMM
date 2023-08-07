#!/bin/bash

APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")

for app1 in "${APPS[@]}"; do
    python src/train.py $app1 ms 0
    python src/1_mm.py $app1 ms 0
done
