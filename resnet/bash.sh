#!/bin/bash


python train.py list_train 75

python predict.py fold0
mv pred.txt pred.fold0

python predict.py fold1
mv pred.txt pred.fold1

python predict.py fold2
mv pred.txt pred.fold2

python predict.py fold3
mv pred.txt pred.fold3

python predict.py fold4
mv pred.txt pred.fold4
