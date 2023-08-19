# MAP_AMM
Memory access prediction using table approximate matrix multiplication: MLP, ResNet.

Change trace dir in train.py, then:
1. run `train.py` in `map` folder to train a model and ouput dataset
2. run `2_resnet.py` in `amm` folder to train and evaluate PQ table approximation

Updates:

1. 2_resnet.py: whole file
2. r_amm.py: a manual and lut implementation of resnet-tiny, as a class
3. pq_amm_cnn.py: pq class for cnn
4. vquantizers.py: PQEncoder_CNN class, _fit_pq_lut_cnn function, 

