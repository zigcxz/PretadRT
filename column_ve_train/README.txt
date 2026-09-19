
运行:

python train_gradient_AE.py


输入:
F:\RepoRT-master (2)\Processed\origin\original_gradient


每个文件只读取:

t [min]
A [%]
B [%]
flow rate [ml/min]


训练:
gradient -> AutoEncoder


输出:

processed/processed/models/gradient_AE.pth

processed/gradient_vector/*.csv


每个gradient输出32维embedding
