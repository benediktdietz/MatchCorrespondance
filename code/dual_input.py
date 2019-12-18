from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import plotly.figure_factory as FF
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import test_dataloaderBD
# import loadTrainDataBD
import argparse
import sys
import os
# from PIL import Image
import tensorflow as tf
import csv
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.preprocessing import maxabs_scale, StandardScaler
import torch
from tensorflow.python import pywrap_tensorflow
import h5py

#########################
#########################
DATA_DIR = 'tensorflow_logs/monday/margin25/'
iterations = 10000
batch_size = 4
window_size = 30
L2margin = 25.
test_size = 100
acc_print_freq = 50
moving_avg_window = 100
what = 'quick_frame30'
zero = tf.constant(1e-04)
#########################
#########################

'''load test data'''
def test_loader(testfile):

    f = h5py.File(testfile)
    data = f['data']
    labels = f['labels']
    label = np.zeros((data.shape[1], f[labels[0,0]][:].shape[1]))
    data_p1 = np.zeros((data.shape[1], window_size, window_size, window_size))
    data_p2 = np.zeros((data.shape[1], window_size, window_size, window_size))
    for i in range(data.shape[1]):   
        label[i] = f[labels[0,i]][:]
        data_p1[i] = f[f['data'][0,i]]['voxelGridTDF'][:]#.reshape(1,window_size,window_size,window_size,1)
        data_p2[i] = f[f['data'][1,i]]['voxelGridTDF'][:]#.reshape(1,window_size,window_size,window_size,1)
    index0 = np.where(label[:,0] == 0)
    index1 = np.where(label[:,0] == 1)
    return data_p1, data_p2, label

def data_loader(what):

    if what == 'triple_samples':

        samples1 = np.load('samples1.npy')
        samples2 = np.load('samples2.npy')
        samples3 = np.load('samples3.npy')
        samples1 = np.reshape(samples1, (-1, 30, 30, 30, 1))
        samples2 = np.reshape(samples2, (-1, 30, 30, 30, 1))
        samples3 = np.reshape(samples3, (-1, 30, 30, 30, 1))

        return samples1, samples2, samples3

    if what == 'full_frame30':

        data_p1_trunk, data_p2_trunk, label_trunk = test_loader('trunk_test.mat')
        data_p1_leaves, data_p2_leaves, label_leaves = test_loader('leaves_test.mat')
        data_p1_branch, data_p2_branch, label_branch = test_loader('branch_test.mat')
        data_p1_test, data_p2_test, label_test = test_loader('test.mat')
        return data_p1_trunk, data_p2_trunk, label_trunk, data_p1_leaves, data_p2_leaves, label_leaves, data_p1_branch, data_p2_branch, label_branch, data_p1_test, data_p2_test, label_test

    if what == 'full_frame45':

        data_p1_trunk_45, data_p2_trunk_45, label_trunk_45 = test_loader('trunk_test_45.mat')
        data_p1_leaves_45, data_p2_leaves_45, label_leaves_45 = test_loader('leaves_test_45.mat')
        data_p1_branch_45, data_p2_branch_45, label_branch_45 = test_loader('branch_test_45.mat')
        data_p1_test_45, data_p2_test_45, label_test_45 = test_loader('test_45.mat')
        return data_p1_trunk_45, data_p2_trunk_45, label_trunk_45, data_p1_leaves_45, data_p2_leaves_45, label_leaves_45, data_p1_branch_45, data_p2_branch_45, label_branch_45, data_p1_test_45, data_p2_test_45, label_test_45

    if what == 'quick_frame30':
        print('loading trunks')
        trunk1 = np.load('trunk1.npy')
        trunk2 = np.load('trunk2.npy')
        trunk3 = np.load('trunk3.npy')
        print('loading leaves')
        leaves1 = np.load('leaves1.npy')
        leaves2 = np.load('leaves2.npy')
        leaves3 = np.load('leaves3.npy')
        print('loading branches')
        branch1 = np.load('branches1.npy')
        branch2 = np.load('branches2.npy')
        branch3 = np.load('branches3.npy')
        print('loading test set')
        test1 = np.load('test1.npy')
        test2 = np.load('test2.npy')
        test3 = np.load('test3.npy')
        return trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3

    if what == 'quick_frame45':
        print('loading trunks')
        trunk1 = np.load('trunk1_45.npy')
        trunk2 = np.load('trunk2_45.npy')
        trunk3 = np.load('trunk3_45.npy')
        print('loading leaves')
        leaves1 = np.load('leaves1_45.npy')
        leaves2 = np.load('leaves2_45.npy')
        leaves3 = np.load('leaves3_45.npy')
        print('loading branches')
        branch1 = np.load('branches1_45.npy')
        branch2 = np.load('branches2_45.npy')
        branch3 = np.load('branches3_45.npy')
        print('loading test set')
        test1 = np.load('test1_45.npy')
        test2 = np.load('test2_45.npy')
        test3 = np.load('test3_45.npy')
        return trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3

def data_prep(trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3):

    print('preparing data...')

    x1 = np.append(trunk1, leaves1, axis=0)
    x1 = np.append(x1, branch1, axis=0)
    x1 = np.append(x1, test1, axis=0)

    x2 = np.append(trunk2, leaves2, axis=0)
    x2 = np.append(x2, branch2, axis=0)
    x2 = np.append(x2, test2, axis=0)

    x3 = np.append(trunk3, leaves3, axis=0)
    x3 = np.append(x3, branch3, axis=0)
    x3 = np.append(x3, test3, axis=0)

    num_samples = x1.shape[0]

    idx = np.random.permutation(num_samples-1)

    idx_train = idx[:np.int(0.8*num_samples)]
    idx_val = idx[np.int(0.8*num_samples)+1:num_samples-1]

    x1_train = x1[idx_train,:]
    x2_train = x2[idx_train,:]
    x3_train = x3[idx_train,:]
    x1_val = x1[idx_val,:]
    x2_val = x2[idx_val,:]
    x3_val = x3[idx_val,:]

    return x1_train, x2_train, x3_train, x1_val, x2_val, x3_val

trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3 = data_loader(what)

x1_train, x2_train, x3_train, x1_val, x2_val, x3_val = data_prep(trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3)

x1_val = np.reshape(x1_val, (-1, window_size, window_size, window_size, 1))
x2_val = np.reshape(x2_val, (-1, window_size, window_size, window_size, 1))

trunk1 = np.reshape(trunk1, (-1, window_size, window_size, window_size, 1))[:test_size,:]
trunk2 = np.reshape(trunk2, (-1, window_size, window_size, window_size, 1))[:test_size,:]
trunk3 = np.reshape(trunk3, (-1, 1))[:test_size,:]
leaves1 = np.reshape(leaves1, (-1, window_size, window_size, window_size, 1))[:test_size,:]
leaves2 = np.reshape(leaves2, (-1, window_size, window_size, window_size, 1))[:test_size,:]
leaves3 = np.reshape(leaves3, (-1, 1))[:test_size,:]
branch1 = np.reshape(branch1, (-1, window_size, window_size, window_size, 1))[:test_size,:]
branch2 = np.reshape(branch2, (-1, window_size, window_size, window_size, 1))[:test_size,:]
branch3 = np.reshape(branch3, (-1, 1))[:test_size,:]
test1 = np.reshape(test1, (-1, window_size, window_size, window_size, 1))[:test_size,:]
test2 = np.reshape(test2, (-1, window_size, window_size, window_size, 1))[:test_size,:]
test3 = np.reshape(test3, (-1, 1))[:test_size,:]

batch_size_tot = x1_train.shape[0]
num_features = np.reshape(x1_train, (batch_size_tot, -1)).shape[1]
# test_size = x1_val.shape[0]
print('total batch size.....................', batch_size_tot)
print('num features.........................', num_features)

 
def variable_summaries(var):
        with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)


def performance(distances, labels, threshold):

    norm = len(distances)

    mistakes = 0

    for i in range(0, norm):

        if labels[i] == 0:

            if distances[i] > threshold:

                mistakes += 1

        if labels[i] == 1:

            if distances[i] < threshold:

                mistakes += 1

    return 1- (mistakes/norm)


def supply_data():

    rands = np.random.randint(0, batch_size_tot-1, batch_size)

    x1 = x1_train[rands,:,:,:]
    x1 = np.reshape(x1, (batch_size, window_size, window_size, window_size, 1))
    x2 = x2_train[rands,:,:,:]
    x2 = np.reshape(x2, (batch_size, window_size, window_size, window_size, 1))

    labels = x3_train[rands]

    return x1, x2, labels


def siamese(input, reuse=False):

    '''Define network structure'''

    with tf.name_scope("siameseNN"):
        
        with tf.variable_scope('conv1') as scope:

            conv1 = tf.layers.conv3d(
                input,
                64,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )


        with tf.variable_scope('conv2') as scope:

            conv2 = tf.layers.conv3d(
                conv1,
                64,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.truncated_normal_initializer(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )

            conv2 = tf.contrib.layers.max_pool3d(
                conv2,
                [2, 2, 2],
                stride=2,
                padding='VALID',
                # data_format=DATA_FORMAT_NDHWC,
                outputs_collections=None,
                scope=None
            )


        with tf.variable_scope('conv3') as scope:

            conv3 = tf.layers.conv3d(
                conv2,
                128,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )


        with tf.variable_scope('conv4') as scope:

            conv4 = tf.layers.conv3d(
                conv3,
                128,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )

            conv4 = tf.contrib.layers.max_pool3d(
                conv4,
                [2, 2, 2],
                stride=2,
                padding='VALID',
                # data_format=DATA_FORMAT_NDHWC,
                outputs_collections=None,
                scope=None
            )


        with tf.variable_scope('conv5') as scope:

            conv5 = tf.layers.conv3d(
                conv4,
                256,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )


        with tf.variable_scope('conv6') as scope:

            conv6 = tf.layers.conv3d(
                conv5,
                256,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )


        with tf.variable_scope('conv7') as scope:
            conv7 = tf.layers.conv3d(
                conv6,
                512,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )


        with tf.variable_scope('conv8') as scope:
            conv8 = tf.layers.conv3d(
                conv7,
                512,
                [3, 3, 3],
                strides=(1, 1, 1),
                padding='SAME',
                # weights_initializer=tf.contrib.layers.xavier_initializer_conv3d(),
                data_format='channels_last',
                dilation_rate=(1, 1, 1),
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                # scope=scope,
                name=None,
                reuse=reuse
            )


        output = tf.contrib.layers.flatten(conv8)

        output = tf.nn.l2_normalize(
            output,
            axis=1,
            epsilon=1e-12,
            name=None)
            # dim=None
        

        # mean, variance = tf.nn.moments(
        #     output,
        #     0,
        #     shift=None,
        #     name=None,
        #     keep_dims=False
        # )

        # output = tf.nn.batch_normalization(
        #     output,
        #     mean,
        #     variance,
        #     offset=None,
        #     scale=None,
        #     variance_epsilon=1e-06
        # )
        # output = tf.contrib.layers.batch_norm(
        #     output,
        #     decay=0.999,
        #     center=True,
        #     scale=False,
        #     epsilon=0.001,
        #     activation_fn=None,
        #     updates_collections=tf.GraphKeys.UPDATE_OPS,
        #     is_training=True)
    
    return output


def contrastive_loss(output_1, output_2, labels):
 
    with tf.name_scope("contrastive_loss"):

        with tf.name_scope('constants'):

            margin = tf.constant(L2margin)
            one = tf.ones(
                [batch_size, 1],
                dtype=tf.float32)  
            zero = tf.constant(1e-06)
            nf =tf.constant(batch_size, tf.float32)

        with tf.name_scope('euclidean_distances'):

            l2distance = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(tf.reshape(output_1, [batch_size, -1]), tf.reshape(output_2, [batch_size, -1])),2), 1)) * tf.constant(1e+05)# + tf.constant(1.)

            l2diff_match = tf.reduce_sum(
                tf.reshape(labels, [1, -1]) * tf.reshape(l2distance, [1, -1])
                ) + zero

            l2diff_non_match = tf.reduce_sum(
                tf.reshape((1. - labels), [1, -1]) * tf.reshape(l2distance, [1, -1])
                ) + zero

            l2diff_match = tf.divide(l2diff_match, (tf.reduce_sum(tf.reshape(labels, [-1]) + zero)))
            l2diff_non_match = tf.divide(l2diff_non_match, (nf  + zero - tf.reduce_sum(tf.reshape(labels, [-1]))))

            # margin = 2 * (tf.reduce_mean(l2distance) + 1.)

        with tf.name_scope('match_loss'):

            part1 = l2diff_match

        with tf.name_scope('non_match_loss'):
         
            partmax = tf.maximum(
                zero,
                tf.subtract(margin, l2diff_non_match))

            part2 = tf.pow(partmax, 2)

        with tf.name_scope('regularization'):

            reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)

            reg_constant = 1e-10
            regularization = reg_constant * sum(reg_losses)

        with tf.name_scope('final_loss'):

            loss_non_reg = np.add(part1, part2)
            # loss = np.add(loss_non_reg, regularization)
            loss = loss_non_reg

        with tf.name_scope('monitor'):

            prd_match = tf.less_equal(l2distance - margin, 0.)
            prd_non_match = tf.greater(l2distance - margin, 0.)

            true_match = tf.equal(labels, 1.)
            true_non_match = tf.equal(labels, 0.)

            false_matches = tf.logical_and(prd_match, true_non_match)
            false_non_matches = tf.logical_and(prd_non_match, true_match)

            false_matches_rate = tf.reduce_mean(tf.cast(false_matches, tf.float32))
            false_non_matches_rate = tf.reduce_mean(tf.cast(false_non_matches, tf.float32))

            accuracy = tf.divide(
                tf.add(
                    tf.reduce_sum(tf.cast(false_matches, tf.float32)),
                    tf.reduce_sum(tf.cast(false_non_matches, tf.float32))), 
                (2 * nf))

            threshold = .5 * (l2diff_match + l2diff_non_match)


        tf.summary.scalar('l2diff_match', tf.reshape(l2diff_match, []))
        tf.summary.scalar('l2diff_non_match', tf.reshape(l2diff_non_match, []))
        tf.summary.scalar('safety_margin', tf.subtract(l2diff_non_match, l2diff_match))
        # tf.summary.scalar('l2distance', tf.reshape(tf.reduce_mean(l2distance), []))
        tf.summary.scalar('match_loss', tf.reshape(part1, []))
        tf.summary.scalar('non_match_loss', tf.reshape(part2, []))
        tf.summary.scalar('regularization', tf.reshape(regularization, []))
        tf.summary.scalar('loss', tf.reshape(loss, []))
        tf.summary.scalar('threshold', tf.reshape(threshold, []))
        # tf.summary.scalar('distance', tf.reshape(tf.divide(tf.reduce_sum(l2distance), nf), []))
        tf.summary.scalar('margin', tf.reshape(margin, []))
        tf.summary.scalar('accuracy', tf.reshape(accuracy, []))
        tf.summary.scalar('false_matches_rate', tf.reshape(false_matches_rate, []))
        tf.summary.scalar('false_non_matches_rate', tf.reshape(false_non_matches_rate, []))


        return loss, l2diff_match, l2diff_non_match, threshold


def dist(output1, output2):

    with tf.name_scope('distance_comp'):

        distances =  tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(tf.reshape(output1, [test_size, -1]), tf.reshape(output2, [test_size, -1])),2), 1)) * tf.constant(1e+05)

        return tf.reshape(distances, [-1])

def specific_performance(output1_test, output2_test):

    distances = dist(output1_test, output2_test)

    print(distances.shape)

    return distances

def monitor_performance(test_dist, labels, threshold):

    count = 0.
    num = labels.shape[0]

    # threshold = np.mean(labels)

    for i in range(0, num):

        if labels[i] == 1:

            if test_dist[i] > threshold:

                count += 1.

        if labels[i] == 0:

            if test_dist[i] < threshold:

                count += 1.

    print('count', count)
    print('num', num)

    return count/num



x1 = tf.placeholder(
    tf.float32, 
    [batch_size, window_size, window_size, window_size, 1],
    name="input1")

x2 = tf.placeholder(
    tf.float32, 
    [batch_size, window_size, window_size, window_size, 1],
    name="input2")

labels = tf.placeholder(
    tf.float32, 
    [batch_size, 1],
    name="input3")


x1_test = tf.placeholder(
    tf.float32, 
    [test_size, window_size, window_size, window_size, 1],
    name="input1_test")

x2_test = tf.placeholder(
    tf.float32, 
    [test_size, window_size, window_size, window_size, 1],
    name="input2_test")

# labels_test = tf.placeholder(
#     tf.float32, 
#     [test_size, 1],
#     name="input3_test")


output1 = siamese(x1, reuse=False)
output2 = siamese(x2, reuse=True)
# output3 = siamese(x3, reuse=True)
output1_test = siamese(x1_test, reuse=True)
output2_test = siamese(x2_test, reuse=True)

test_distances = dist(output1_test, output2_test)

# test_distances = dist(output1_test, output2_test)


# tf.summary.histogram('output1', output1)
# tf.summary.histogram('output2', output2)

loss, l2diff_match, l2diff_non_match, threshold = contrastive_loss(output1, output2, labels)


# train_step = tf.train.GradientDescentOptimizer(1e-02).minimize(loss)

# train_step = tf.train.AdamOptimizer(
#     learning_rate=0.001,
#     beta1=0.9,
#     beta2=0.999,
#     epsilon=1e-08,
#     use_locking=False,
#     name='Adam'
#     ).minimize(loss)

 
merged = tf.summary.merge_all()



threshold_history = np.zeros(iterations)

def moving_threshold(threshold_history, i):

    if i == 0:

        return L2margin
    elif i < 50:

        return np.mean(threshold_history[:i])

    elif i >= 50:

        return np.mean(threshold_history[i - moving_avg_window:i])




with tf.Session() as sess:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_
        train_step = tf.train.AdamOptimizer(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            use_locking=False,
            name='Adam'
            ).minimize(loss)

    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(
        DATA_DIR,
        graph = sess.graph)


    # print("## Trainable variables: ")
    # for v in tf.trainable_variables():
    #     print(v.name)

    # print("## update variables: ")
    # for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
    #     print(v.name)


    for i in range(0,iterations):

        data1, data2, data3 = supply_data()

        summary, _ , loss_train, l2md, l2nonmd, thresh = sess.run([
            merged, 
            train_step, 
            loss, 
            l2diff_match, 
            l2diff_non_match, 
            threshold
            ], 
            feed_dict={
                x1: data1,
                x2: data2,
                labels: data3
                })


        threshold_history[i] = thresh

        test_threshold = moving_threshold(threshold_history, i)


        if i % 2 == 0:

            print('############################################## iteration ', i)
            print('@', DATA_DIR, '| margin=', L2margin, '| batch=', batch_size)
            print('##############################################')
            print('loss...................................', loss_train)
            print('l2diff_match...........................', l2md)
            print('l2diff_non_match.......................', l2nonmd)
            print('threshold..............................', thresh)
            # print(test_dist)
            # print(trunk3)

        if i % acc_print_freq == 0:

            test_dist_trunk = sess.run(test_distances, feed_dict={
                x1_test: trunk1,
                x2_test: trunk2})

            trunk_acc =  monitor_performance(test_dist_trunk, trunk3, L2margin)
            trunk_acc_t =  monitor_performance(test_dist_trunk, trunk3, test_threshold)
            print('tunk performance---------------------------------------', trunk_acc)
            print('tunk performance dynamic threshold---------------------', trunk_acc_t)


            test_dist_branch = sess.run(test_distances, feed_dict={
                x1_test: branch1,
                x2_test: branch2})

            branch_acc =  monitor_performance(test_dist_branch, branch3, L2margin)
            branch_acc_t =  monitor_performance(test_dist_branch, branch3, test_threshold)
            print('branch performance---------------------------------------', branch_acc)
            print('branch performance dynamic threshold---------------------', branch_acc_t)


            test_dist_leaves = sess.run(test_distances, feed_dict={
                x1_test: leaves1,
                x2_test: leaves2})

            leaves_acc =  monitor_performance(test_dist_leaves, leaves3, L2margin)
            leaves_acc_t =  monitor_performance(test_dist_leaves, leaves3, test_threshold)
            print('leaves performance---------------------------------------', leaves_acc)
            print('leaves performance dynamic threshold---------------------', leaves_acc_t)

            test_dist = sess.run(test_distances, feed_dict={
                x1_test: test1,
                x2_test: test2})

            test_acc =  monitor_performance(test_dist, test3, L2margin)
            test_acc_t =  monitor_performance(test_dist, test3, test_threshold)
            print('test performance---------------------------------------', test_acc)
            print('test performance dynamic threshold---------------------', test_acc_t)

        if i == 0:

            print('input shape............................', data1.shape)

            with tf.variable_scope('conv1', reuse=True):
                conv1_kernel = tf.get_variable('conv3d/kernel')
                conv1_kernel = tf.reshape(conv1_kernel, [1, -1])

                # if i == 0:
                #     conv1_cache = np.zeros((int(iterations/100), conv1_kernel.shape[1]))

                # if i % 100 == 0:
                #     conv1_cache[i,:] = conv1_kernel.eval()

                print('conv1_kernel---------', conv1_kernel.shape)

            with tf.variable_scope('conv2', reuse=True):
                conv2_kernel = tf.get_variable('conv3d/kernel')
                conv2_kernel = tf.reshape(conv2_kernel, [1, -1])

                # if i == 0:
                #     conv2_cache = np.zeros((int(iterations/100), conv2_kernel.shape[1]))

                # if i % 100 == 0:
                #     conv2_cache[i,:] = conv2_kernel.eval()

                print('conv2_kernel---------', conv2_kernel.shape)

            with tf.variable_scope('conv3', reuse=True):
                conv3_kernel = tf.get_variable('conv3d/kernel')
                conv3_kernel = tf.reshape(conv3_kernel, [1, -1])

            print('conv3_kernel---------', conv3_kernel.shape)

            with tf.variable_scope('conv4', reuse=True):
                conv4_kernel = tf.get_variable('conv3d/kernel')
                conv4_kernel = tf.reshape(conv4_kernel, [1, -1])

            print('conv4_kernel---------', conv4_kernel.shape)

            with tf.variable_scope('conv5', reuse=True):
                conv5_kernel = tf.get_variable('conv3d/kernel')
                conv5_kernel = tf.reshape(conv5_kernel, [1, -1])

                # if i == 0:
                #     conv5_cache = np.zeros((int(iterations/100), conv5_kernel.shape[1]))

                # if i % 100 == 0:
                #     conv5_cache[i,:] = conv5_kernel.eval()

                print('conv5_kernel---------', conv5_kernel.shape)

            with tf.variable_scope('conv6', reuse=True):
                conv6_kernel = tf.get_variable('conv3d/kernel')
                conv6_kernel = tf.reshape(conv6_kernel, [1, -1])

            print('conv6_kernel---------', conv6_kernel.shape)

            with tf.variable_scope('conv7', reuse=True):
                conv7_kernel = tf.get_variable('conv3d/kernel')
                conv7_kernel = tf.reshape(conv7_kernel, [1, -1])

            print('conv7_kernel---------', conv7_kernel.shape)

            with tf.variable_scope('conv8', reuse=True):
                conv8_kernel = tf.get_variable('conv3d/kernel')
                conv8_kernel = tf.reshape(conv8_kernel, [1, -1])

            print('conv8_kernel---------', conv8_kernel.shape)

        
        # multi_slice_viewer(np.reshape(out, (batch_size, dummy, dummy, dummy, 1))[0,:,:,:,0])

        train_writer.add_summary(summary, i)

        if np.isnan(loss_train):
            print('Model diverged with loss = NaN')
            quit()


# print(w_1)