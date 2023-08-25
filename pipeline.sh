#python src/preprocess.py bc-3.txt.xz 4 
python src/train.py bc-3.txt.xz ms 4 
python src/train.py bc-3.txt.xz mt 4 
python src/train_kd.py bc-3.txt.xz mt ms 4 
python src/1_mm.py ms.stu 0 256,256,256 1,1,1
python src/generate.py bc-3.txt.xz ms 4 
python src/generate.py bc-3.txt.xz ms.stu 4 
