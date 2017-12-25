#!/usr/bin/env python

import random
import cv2
import numpy as np

def GS_split_224_resize (FILE,size):
    all_train_line=[]
    TRAIN_LIST=open(FILE,'r')
    for line in TRAIN_LIST:
        line=line.rstrip()
        all_train_line.append(line)

    random.shuffle(all_train_line)
    
    i=0
    fold0_input= [] 
    fold0_label= []
    fold1_input= [] 
    fold1_label= []
    fold2_input= [] 
    fold2_label= []
    fold3_input= [] 
    fold3_label= []
    fold4_input= [] 
    fold4_label= []
    cut1=int(len(all_train_line)/5)
    cut2=int(len(all_train_line)/5*2)
    cut3=int(len(all_train_line)/5*3)
    cut4=int(len(all_train_line)/5*4)
    for line in all_train_line:
        table=line.split('\t')
        img= np.load(table[0])
        img= cv2.resize(img,(224,224))
        band2=table[0].replace('band_1','band_2')
        img2= np.load(band2)
        img2= cv2.resize(img2,(224,224))
        img=np.stack((img,img2),axis=2)
        label=table[1]
        if (i<cut1):
            fold0_input.append(img)
            fold0_label.append(label)
        elif (i<cut2):
            fold1_input.append(img)
            fold1_label.append(label)
        elif (i<cut3):
            fold2_input.append(img)
            fold2_label.append(label)
        elif (i<cut4):
            fold3_input.append(img)
            fold3_label.append(label)
        else:
            fold4_input.append(img)
            fold4_label.append(label)

        i=i+1

    fold0_input=np.asarray(fold0_input)
    fold0_label=np.asarray(fold0_label)

    fold1_input=np.asarray(fold1_input)
    fold1_label=np.asarray(fold1_label)

    fold2_input=np.asarray(fold2_input)
    fold2_label=np.asarray(fold2_label)

    fold3_input=np.asarray(fold3_input)
    fold3_label=np.asarray(fold3_label)

    fold4_input=np.asarray(fold4_input)
    fold4_label=np.asarray(fold4_label)

    return (fold0_input, fold0_label, fold1_input, fold1_label, fold2_input, fold2_label, fold3_input, fold3_label, fold4_input, fold4_label)
