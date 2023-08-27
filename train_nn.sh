APPS=("410.bwaves-s0.txt.xz" "602.gcc-s0.txt.xz" "bc-3.txt.xz")

for app1 in "${APPS[@]}"; do
    #python src/preprocess.py $app1 4
    python src/train.py $app1 rs 1 
done
