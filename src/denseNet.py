# -*- coding: utf-8 -*-

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm,flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import learn
import numpy as np
import os

growth_rate = 32
filter = 64
nb_block = 3
dropout_rate = 0.5


def conv_layer(input,filter,kernel,stride=1,layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input,use_bias=False,filters=filter,kernel_size=kernel,strides=stride,padding='SAME')
        return network

def Global_Average_Pooling(x,stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width,height]
    return tf.layer.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride)
    # The strdie value does not matter.It is global average pooling without tflearn
    """
    return global_avg_pool(x,name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not

def Batch_Normalization(x,training,scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   update_collections=None,
                   deacy=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        if training:
            return tf.layers.batch_normalization(x,axis=3,training=training,reuse=True)

        else:
            return tf.layers.batch_normalization(x,axis=3,training=training,reuse=True)


# def Batch_Normalization(x, training, scope):
#     with arg_scope([batch_norm],
#                    scope=scope,
#                    updates_collections=None,
#                    decay=0.9,
#                    center=True,
#                    scale=True,
#                    zero_debias_moving_mean=True) :
#         return tf.cond(training,
#                        lambda : batch_norm(inputs=x, is_training=training, reuse=None),
#                        lambda : batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x,rate,training):
    return tf.layers.dropout(inputs=x,rate=rate,training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x,pool_size=[2,2],stride=2,padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Max_Pooling(x,pool_size=[3,3],stride=2,padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Concatenation(layers):
    return tf.concat(layers,axis=3)



def bottleneck_layer(x,scope,training):
    with tf.name_scope(scope):
        x = Batch_Normalization(x,training=training,scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x,filter=4*growth_rate,kernel=[1,1],layer_name=scope+'_conv1')
        x = Drop_out(x,rate=dropout_rate,training=training)

        x =Batch_Normalization(x,training=training,scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x,filter=growth_rate,kernel=[3,3],layer_name=scope+'_conv2')
        x = Drop_out(x,rate=dropout_rate,training=training)
        return x

def transition_layer(x,filters,scope,training):
    with tf.name_scope(scope):
        x = Batch_Normalization(x,training=training,scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x,filter = filters,kernel=[1,1],layer_name=scope+'_conv1')
        x = Drop_out(x,rate=dropout_rate,training=training)
        x = Average_pooling(x,pool_size=[2,2],stride=[2,1])
        return x

def dense_block(input_x,nb_layers,layer_name,training):
    with tf.name_scope(layer_name):
        layers_concat =list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x,scope=layer_name+'_bottleN_'+str(0),training=training)

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x,scope=layer_name + '_bottleN_' + str(i+1),training=training)
            layers_concat.append(x)

        x = Concatenation(layers_concat)

        return x

def Dense_net(input_x,widths,mode):

    training = (mode == learn.ModeKeys.TRAIN)
    # input_x:[ 32 ,width , 3 ]
    x = conv_layer(input_x,filter=filter,kernel=[3,3],stride=1,layer_name='conv0')
    # x = Max_Pooling(x,pool_size=[3,3],stride=2)
    # x: [32,width,64]
    x = dense_block(input_x = x,nb_layers=4,layer_name='dense_1',training=training)
    # x: [32,width,64+4*32=192]
    x = transition_layer(x,128,scope='trans_1',training=training)#transition_layer(x,filters,scope,training)
    # x: [16,width-1,128]
    x = dense_block(input_x = x,nb_layers=6,layer_name='dense_2',training=training)
    # x: [16,width,128+6*32=320]
    x = transition_layer(x,256,scope='trans_2',training=training)
    # x: [8,width-1,256]
    x = Max_Pooling(x,[2,2],2)
    # x:[4,width/2,256]
    x = dense_block(input_x =x ,nb_layers=8,layer_name='dense_3',training=training)
    # x: [4,width,256+8*32=512]
    x = transition_layer(x,512,scope='trans_3',training=training)
    # x: [4,width-1,512]

    x = Batch_Normalization(x,training=training,scope='linear_batch')
    x = Relu(x)
    # x = Global_Average_Pooling(x)  # cifar-10中用于分类
    x = Max_Pooling(x,[2,2],[2,1])
    # x: [1,width/2,512]

    features = tf.squeeze(x,axis=1,name='features')
    # calculate resulting sequence length
    one = tf.constant(1, dtype=tf.int32, name='one')
    two = tf.constant(2, dtype=tf.int32, name='two')

    after_conv0=widths
    after_dense_1=after_conv0
    after_trans_1=tf.subtract(after_dense_1,one)
    after_dense_2=after_trans_1
    after_trans_2=tf.subtract(after_dense_2,one)
    after_first_maxpool=tf.floor_div(after_trans_2, two )#向下取整
    after_dense_3=after_first_maxpool
    after_trans_3=tf.subtract(after_dense_3,one)
    after_second_maxpool=tf.subtract(after_trans_3,one)
    sequence_length = tf.reshape(after_second_maxpool,[-1], name='seq_len')

    return features,sequence_length




