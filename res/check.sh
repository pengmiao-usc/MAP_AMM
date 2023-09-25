#!/bin/bash

APPS=("410.bwaves-s0" "433.milc-s0" "437.leslie3d-s0" "462.libquantum-s0" "602.gcc-s0" "605.mcf-s0" "619.lbm-s0" "621.wrf-s2")
# APPS=("410.bwaves-s0" "433.milc-s0")

for app in ${APPS[@]}
do
    FILES=("$app.vitt.pkl.val_res.json" "$app.vit.pkl.val_res.json" "$app.vit.stu.75.0.pkl.val_res.json")

    for file in ${FILES[@]}
    do
        if [[ -e $file ]]; then
            cat $file | grep "f1"
        else
            echo "File $file does not exist!"
        fi
    done
done

