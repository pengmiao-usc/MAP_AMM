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

## Teacher Model Abbreviations (No corresponding AMM file)
|option|model   |
|------|--------|
|mst   |MLPDemo |
|rst   |ResNet  |
|vitt  |TMAP    |
|mixt  |MLPMixer|

## Workflow
- Change directories and hyperparameters in `params.yaml`
- Preprocss trace using `python src/preprocess.py {trace} {gpu no.}`
- Train NN using `python src/train.py {trace} {option} {gpu no.}` 
- Generate NN prefetch file using `python src/generate.py {trace} {option} {gpu no.}`
- Train AMM w/o Fine-Tuning `taskset -c {core 0}-{core 63} python src/{amm file} {trace} {corresponding option} K,...,K N,...,N {gpu no.}`
- Generate AMM prefetch file using `python src/generate_amm.py {trace} {option} K,...,K N,...,N {gpu no.}`

## KD Workflow
- Train Teacher Model: `python src/train.py {trace} {option}t {gpu no.}`
- Train Student Model via KD: `python src/train_kd.py {trace} {teacher model} {student model} {gpu no.} {alpha 0 - 1} {temperature 1 - 5}` 
- Find outputs with the names of `{app}.{option}.stu.a.{alpha}.t.{temp}` in `/res` and `/model`
