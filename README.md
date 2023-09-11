# TNN-Fetch
Memory access prediction using table approximated matrix multiplication. 

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
|Baseline structure|N/K count|
|------------------|---------|
|MLP Demo          |3        |
|ResNet_Tiny       |5        |
|MLP Mixer         |14       |
|ViT               |14       |

## Model Abbreviations
|option|model   |amm file   |
|------|--------|-----------|
|ms    |MLPDemo |1_mlp.py   |
|rs    |ResNet  |2_resnet.py|
|vit   |TMAP    |3_vit.py   |
|mix   |MLPMixer|4_mixer.py |

## Workflow
- Change directories and hyperparameters in `params.yaml`
- Preprocss trace using `python src/preprocess.py {app}.txt.xz {gpu no.}`
- Train NN using `python src/train.py {app}.txt.xz {option} {gpu no.}`
- Generate NN prefetch file using `python src/generate.py {app.txt.xz} {option} {gpu no.}`
- Train AMM using `python src/{amm file} {app}.txt.xz {corresponding option} K,...,K N,...,N {gpu no.}`
- Generate AMM prefetch file using `python src/generate_amm.py {app}.txt.xz {option} K,...,K N,...,N {gpu no.}`