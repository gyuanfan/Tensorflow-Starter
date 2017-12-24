#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pkgutil
import numpy as np
import os
import datetime
import cv2
#import picpac
import tensorflow as tf
import tensorflow.contrib.slim as slim
import GS_split_299_resize
import sys
import random
import scipy.ndimage.interpolation
from scipy.ndimage import zoom

size=int(sys.argv[2])
max_epoch=200
batch=16
max_rotate=0
max_shift=60
max_scale=1.2
min_scale=0.9


def shift (img,shift_x,shift_y,the_min):
    (x,y)=img.shape
    newimg=np.ones((x,y))*the_min
    if (shift_x>0):
        newimg[shift_x:x,:]=img[0:(x-shift_x),:]
    else:
        newimg[0:(x+shift_x),:]=img[-shift_x:x,:]

    img=newimg
    if (shift_y>0):
        newimg[:,shift_y:y]=img[:,0:(y-shift_y)]
    else:
        newimg[:,0:(y+shift_y)]=img[:,-shift_y:y]
    return newimg

def scaleImage (image,scale):
    [x,y]= image.shape
    x1=int(round(x*scale))
    y1=int(round(y*scale))
    image=cv2.resize(image,(y1,x1))
    new=np.zeros((x,y))
    if (x1>x):
        start=int(round(x1/2-x/2))
        end=start+x
        new=image[start:end,start:end]
    else:
        new_start=int(round(x-x1)/2)
        new_end=new_start+x1
        new[new_start:new_end,new_start:new_end]=image
    return new



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('opt','adam', '')
flags.DEFINE_string('mixin', None, 'database')
flags.DEFINE_integer('classes', '2', 'number of classes')
flags.DEFINE_integer('resize', None, '')
flags.DEFINE_integer('max_size', None, '')
flags.DEFINE_integer('channels', 2, '')
flags.DEFINE_string('net', 'inception_v3_mod.inception_v3', 'cnn architecture, e.g. vgg.vgg_a')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 100, '')
flags.DEFINE_integer('test_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('save_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('max_steps', 600000, 'Number of steps to run trainer.')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_integer('split', 1, 'split into this number of parts for cross-validation')
flags.DEFINE_integer('split_fold', 1, 'part index for cross-validation')
flags.DEFINE_integer('max_to_keep', 20, '')

# load network architecture by name
def inference (inputs, num_classes):
    full = FLAGS.net
    # e.g. full == 'tensorflow.contrib.slim.python.slim.nets.vgg.vgg_a'
    fs = full.split('.')
    loader = pkgutil.find_loader('.'.join(fs[:-1]))
    module = loader.load_module('')
    net = getattr(module, fs[-1])
    logits, _ = net(inputs, num_classes)
    #logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_int32(labels)    # float from picpac
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        hit = tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32)
        return [tf.reduce_mean(xe, name='xentropy_mean'), tf.reduce_mean(hit, name='accuracy_total')]
    pass

def training (loss, rate):
    #tf.scalar_summary(loss.op.name, loss)
   
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
        #tf.summary.scalar('learning_rate', rate)

    if FLAGS.opt == 'adam':
        rate /= 100
        optimizer = tf.train.AdamOptimizer(rate)
        print('adam!')
    else:
        optimizer = tf.train.GradientDescentOptimizer(rate)
        print('gradient!')
        pass

    return optimizer.minimize(loss, global_step=global_step)

def run_training (train_input_var,train_label_var,test_input_var,test_label_var,fold):
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    if not FLAGS.mixin is None:
        config['mixin'] = FLAGS.mixin
        config['mixin_group_delta'] = 0
    if not FLAGS.max_size is None:
        config['max_size'] = FLAGS.max_size


    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        Y_ = tf.placeholder(tf.float32, shape=(None,), name="labels")
        logits = inference(X, FLAGS.classes)


        loss, accuracy = fcn_loss(logits, Y_)
        train_op = training(loss, FLAGS.learning_rate)
        #summary_op = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, tf.get_default_graph())

        init = tf.global_variables_initializer()

        #graph_txt = tf.get_default_graph().as_graph_def().SerializeToString()
        #with open(os.path.join(FLAGS.train_dir, "graph"), "w") as f:
        #    f.write(graph_txt)
        #    pass

        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        with tf.Session(config=config) as sess:
            sess.run(init)
            total_train_image=0
            initial_test_loss=100
            for epoch  in range(0,max_epoch):
                loss_sum = 0
                accuracy_sum = 0
                batch_sum = 0
                count=int(train_input_var.shape[0]/batch)
                for i in range(0, count):
                    tmp_input=[] 
                    tmp_label=[] 
                    jjj=i*batch
                    while (jjj<(i*batch+batch)):
                        img=train_input_var[jjj,:,:,:]
                        #img1=img[:,:,0].reshape(size,size)
                        #img2=img[:,:,1].reshape(size,size)
                        img1=img[:,:,0]
                        img2=img[:,:,1]

                        ### rotate
                        rrr=random.random()
                        rrr_rotate=rrr*max_rotate
                        img1=scipy.ndimage.interpolation.rotate(img1, rrr_rotate,reshape=False) 
                        img2=scipy.ndimage.interpolation.rotate(img2, rrr_rotate,reshape=False) 

                        ### scale 
                        
                        rrr=random.random()
                        rrr_scale=rrr*(max_scale-min_scale)+min_scale
                        img1=scaleImage(img1,rrr_scale)
                        img2=scaleImage(img2,rrr_scale)



                        rrr=(random.random()-0.5)*2
                        rrr_shift_x=int(np.floor(rrr*max_shift))
                        rrr=(random.random()-0.5)*2
                        rrr_shift_y=int(np.floor(rrr*max_shift))

                        #img1=scipy.ndimage.interpolation.shift(img1, [rrr_shift_x,rrr_shift_y])
                        #img2=scipy.ndimage.interpolation.shift(img2, [rrr_shift_x,rrr_shift_y])
                        img1=shift(img1,rrr_shift_x,rrr_shift_y,img1.min())
                        img2=shift(img2,rrr_shift_x,rrr_shift_y,img2.min())
                        img=np.stack((img1,img2),axis=2)
                
                        tmp_input.append(img)

                        label=train_label_var[jjj]
                        tmp_label.append(label)
                        jjj=jjj+1
                        total_train_image=total_train_image+1

                    images=np.asarray(tmp_input)
                    labels=np.asarray(tmp_label)
                    #print(images.shape, labels.shape)
                    feed_dict = {X: images,
                                 Y_: labels}
                    _, loss_value, accuracy_value, ll = sess.run([train_op, loss, accuracy, logits], feed_dict=feed_dict)


                    loss_sum += loss_value * batch
                    accuracy_sum += accuracy_value * batch
                    batch_sum += batch
                    if (total_train_image ) % 320 == 0:
                        print(datetime.datetime.now())
                        print('i %d: loss = %.4f, accuracy = %.4f' % (i+1, loss_sum/batch_sum, accuracy_sum/batch_sum))
                        loss_sum = 0
                        accuracy_sum = 0
                        batch_sum = 0


                    if (total_train_image ) % 640 == 0:
                        batch_sum2 = 0
                        loss_sum2 = 0
                        accuracy_sum2 = 0
                        count_test=int(test_input_var.shape[0]/batch)
                        for test_i in range(0, count_test):
                            tmp_input=[] 
                            tmp_label=[] 
                            jjj=test_i*batch
                            while (jjj<(test_i*batch+batch)):
                                img=test_input_var[jjj,:,:,:]
                                img1=img[:,:,0].reshape(size,size)
                                img2=img[:,:,1].reshape(size,size)

                                img=np.stack((img1,img2),axis=2)
                
                                tmp_input.append(img)

                                label=test_label_var[jjj]
                                tmp_label.append(label)
                                jjj=jjj+1

                            images=np.asarray(tmp_input)
                            labels=np.asarray(tmp_label)

                            feed_dict = {X: images,
                                         Y_: labels}


                            loss_value, accuracy_value,ll = sess.run([loss, accuracy,logits], feed_dict=feed_dict)
                           

                            batch_sum2 += batch
                            loss_sum2 += loss_value * batch
                            accuracy_sum2 += accuracy_value * batch

                        print('total numer:%d, step:%d' % (batch_sum2, i+1))
                        print('evaluation: loss = %.4f, accuracy = %.4f, total_loss=%.4f, total_num=%.4f' % (loss_sum2/batch_sum2, accuracy_sum2/batch_sum2,loss_sum2,batch_sum2))
                        test_loss=loss_sum2/batch_sum2
                        if (test_loss<initial_test_loss):
                            #ckpt_path = '%s/%s' % (FLAGS.model, (epoch+ 1))
                            ckpt_path = '%s/%s' % (FLAGS.model, (fold))

                            saver.save(sess, ckpt_path)
                            initial_test_loss=test_loss
                    pass
                pass
            pass
        pass
    pass


def main (_):
    (fold0_input, fold0_label, fold1_input, fold1_label, fold2_input, fold2_label, fold3_input, fold3_label, fold4_input, fold4_label)=GS_split_299_resize.GS_split_299_resize(sys.argv[1],size)
    train_input_var=np.vstack((fold0_input,fold1_input,fold2_input,fold3_input))
    train_label_var=np.concatenate((fold0_label,fold1_label,fold2_label,fold3_label))
    test_input_var=fold4_input
    test_label_var=fold4_label
    run_training(train_input_var,train_label_var,test_input_var,test_label_var,'fold4')

    (fold0_input, fold0_label, fold1_input, fold1_label, fold2_input, fold2_label, fold3_input, fold3_label, fold4_input, fold4_label)=GS_split_299_resize.GS_split_299_resize(sys.argv[1],size)
    train_input_var=np.vstack((fold4_input,fold1_input,fold2_input,fold3_input))
    train_label_var=np.concatenate((fold4_label,fold1_label,fold2_label,fold3_label))
    test_input_var=fold0_input
    test_label_var=fold0_label
    run_training(train_input_var,train_label_var,test_input_var,test_label_var,'fold0')

    (fold0_input, fold0_label, fold1_input, fold1_label, fold2_input, fold2_label, fold3_input, fold3_label, fold4_input, fold4_label)=GS_split_299_resize.GS_split_299_resize(sys.argv[1],size)
    train_input_var=np.vstack((fold4_input,fold1_input,fold0_input,fold3_input))
    train_label_var=np.concatenate((fold4_label,fold1_label,fold0_label,fold3_label))
    test_input_var=fold2_input
    test_label_var=fold2_label
    run_training(train_input_var,train_label_var,test_input_var,test_label_var,'fold2')


    (fold0_input, fold0_label, fold1_input, fold1_label, fold2_input, fold2_label, fold3_input, fold3_label, fold4_input, fold4_label)=GS_split_299_resize.GS_split_299_resize(sys.argv[1],size)
    train_input_var=np.vstack((fold4_input,fold1_input,fold0_input,fold2_input))
    train_label_var=np.concatenate((fold4_label,fold1_label,fold0_label,fold2_label))
    test_input_var=fold3_input
    test_label_var=fold3_label
    run_training(train_input_var,train_label_var,test_input_var,test_label_var,'fold3')


    (fold0_input, fold0_label, fold1_input, fold1_label, fold2_input, fold2_label, fold3_input, fold3_label, fold4_input, fold4_label)=GS_split_299_resize.GS_split_299_resize(sys.argv[1],size)
    train_input_var=np.vstack((fold4_input,fold3_input,fold0_input,fold2_input))
    train_label_var=np.concatenate((fold4_label,fold3_label,fold0_label,fold2_label))
    test_input_var=fold1_input
    test_label_var=fold1_label
    run_training(train_input_var,train_label_var,test_input_var,test_label_var,'fold1')

if __name__ == '__main__':
    tf.app.run()

