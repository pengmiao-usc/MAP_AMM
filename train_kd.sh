APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")
N_VAL=(1 2 4 8)

for app1 in "${APPS[@]}"; do
    #python src/preprocess.py $app1 4
    python src/train_kd.py $app1 rt rs 0
    for n in "${N_VALS[@]}"; do
        python src/2_resnet.py $app rs.stu 0 256,256,256,256,256 "$n,$n,$n,$n,$n" 
    done 
done
