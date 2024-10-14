# 随机删除 1/5 的行


import random 
txtFile = "/home/jiahan/jiahan/codes/Depth-Anything-V2/metric_depth/dataset/splits/UCL_aug2/train.txt"
newTxtFile = "/home/jiahan/jiahan/codes/Depth-Anything-V2/metric_depth/dataset/splits/UCL_aug2/train_0.8.txt"

with open(txtFile, 'r') as f:
    lines = f.readlines()
    
newLines = []
for line in lines:
    if random.random() > 0.2:
        newLines.append(line)

with open(newTxtFile, 'w+') as f:
    for line in newLines:
        f.write(line)

