#!/bin/bash

python preprocess.py 602.gcc-s0.txt.xz 0
python train.py 602.gcc-s0.txt.xz ms 0
python 1_mm.py 602.gcc-s0.txt.xz ms 0
