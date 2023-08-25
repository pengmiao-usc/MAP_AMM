# MAP_AMM
Memory access prediction using table approximate matrix multiplication: MLP, ResNet.

<<<<<<< HEAD
## GPU Mapping
|nvidia-smi|gpu no.|
|----------|-------|
|0 (A4000) |4      |
|1 (A4000) |5      |
|2 (A5000) |0      |
|3 (A5000) |1      |
|4 (A5000) |2      |
|5 (A5000) |3      |


## Models
- ms or MLP Simple
- mt or MLP Teacher
- mm or MLP Mixer

Change trace dir in `params.yaml`, then:
1. run `python src/preprocess.py {app} {gpu no.}`
2. run `python src/train.py {app} {model} {gpu no.}`
3.  a. run `python src/1_mm.py {app} ms {gpu no.}`
    b. run `python src/2_mm.py {app} mt {gpu no.}

To use KD, instead of 2 and 3:

2. run `python src/train_kd.py {app} {tch model}/mt {stu model}/ms {gpu no.}`
3. run `python src/1_mm.py {app} ms.stu {gpu no.}`
 
=======
Change trace dir in train.py, then:
1. run `train.py` in `map` folder to train a model and ouput dataset
2. run `2_resnet.py` in `amm` folder to train and evaluate PQ table approximation

Updates:

1. 2_resnet.py: whole file
2. r_amm.py: a manual and lut implementation of resnet-tiny, as a class
3. pq_amm_cnn.py: pq class for cnn
4. vquantizers.py: PQEncoder_CNN class, _fit_pq_lut_cnn function, 

>>>>>>> d71d9a71b09c12b6b9380caa7b15ff5d962b8906
