#!/bin/bash


python train.py list_train 299

python predict_all_vis.py fold0
mv pred.txt pred.fold0

python predict_all_vis.py fold1
mv pred.txt pred.fold1

python predict_all_vis.py fold2
mv pred.txt pred.fold2

python predict_all_vis.py fold3
mv pred.txt pred.fold3

python predict_all_vis.py fold4
mv pred.txt pred.fold4
