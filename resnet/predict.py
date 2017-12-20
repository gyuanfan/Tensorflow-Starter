#!/usr/bin/env python
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import numpy as np
import pkgutil
import cv2
import sys

fold=sys.argv[1]

size=75
X = tf.placeholder(tf.float32, shape=(None, None, None, 2), name="images")
mg = meta_graph.read_meta_graph_file('model/'+fold+'.meta')
logits = tf.import_graph_def(mg.graph_def, name='xxx', input_map={'images:0': X}, return_elements=['logits:0'])
prob = tf.nn.softmax(logits)
saver = tf.train.Saver(saver_def=mg.saver_def, name='xxx')

with tf.Session() as sess:
    saver.restore(sess, 'model/'+fold)
    tmp_input=np.zeros((1,75,75,2))
    TEST=open('list_test','r')
    PRED=open('pred.txt','w')
    for line in TEST:

        table=line.split('\t')

        img= np.load(table[0])
        img= cv2.resize(img,(75,75))
        img= img.reshape(1,75,75)
        band2=table[0].replace('band_1','band_2')
        img2= np.load(band2)
        img2= cv2.resize(img2,(75,75))
        img2= img2.reshape(1,75,75)
        img=np.stack((img,img2),axis=1)
        img=img.reshape(2,75,75)

        img1=img[0,:,:].reshape(size,size)
        img2=img[1,:,:].reshape(size,size)
        img=np.stack((img1,img2),axis=2)

        tmp_input=[]
        tmp_input.append(img)
        images=np.asarray(tmp_input)
        pp = sess.run(prob, feed_dict={X: images})[0][0][1]
        PRED.write(('%.6f\n') % pp)

    #print(sess.run('w1:0'))
