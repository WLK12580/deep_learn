#!/bin/bash
#注意不要忘记修改 models中模型配置文件下的类别，比如：yolov5s.yaml 在Voc训练中将nc改为20
python train.py --data VOC.yaml --weights yolov5s.pt --img 640  
