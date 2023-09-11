APPS=("410.bwaves-s0.txt.xz") 
N_VALS=(1 2 4 8)

for app1 in "${APPS[@]}"; do
    #python src/preprocess.py $app1 4
    #python src/train_kd.py $app1 rt rs 0
    for n in "${N_VALS[@]}"; do
        python src/2_resnet.py "$app1" rs "256,256,256,256,256" "$n,$n,$n,$n,$n" 5 
    done 
done
