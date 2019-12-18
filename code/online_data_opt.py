from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import plotly.figure_factory as FF
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
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
from tensorflow.contrib.tensorboard.plugins import projector

#########################
#########################
DATA_DIR = 'tensorflow_logs/monday/first/'
iterations = 10000
batch_size = 2
batch_size_train = 16
window_size = 30
L2margin = 80.
test_size = 10
acc_print_freq = 50
moving_avg_window = 100
test_multiplier = 25
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
        trunk1 = np.load('saved_data/trunk1.npy')
        trunk2 = np.load('saved_data/trunk2.npy')
        trunk3 = np.load('saved_data/trunk3.npy')
        print('loading leaves')
        leaves1 = np.load('saved_data/leaves1.npy')
        leaves2 = np.load('saved_data/leaves2.npy')
        leaves3 = np.load('saved_data/leaves3.npy')
        print('loading branches')
        branch1 = np.load('saved_data/branches1.npy')
        branch2 = np.load('saved_data/branches2.npy')
        branch3 = np.load('saved_data/branches3.npy')
        print('loading test set')
        test1 = np.load('saved_data/test1.npy')
        test2 = np.load('saved_data/test2.npy')
        test3 = np.load('saved_data/test3.npy')
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

def data_prep2(trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3):

    print('preparing data...')

    num_samples = trunk1.shape[0]

    idx = np.random.permutation(num_samples-1)

    idx_train = idx[:np.int(0.8*num_samples)]
    idx_val = idx[np.int(0.8*num_samples)+1:num_samples-1]

    x1_train = np.append(trunk1[idx_train,:], leaves1[idx_train,:], axis=0)
    x1_train = np.append(x1_train, branch1[idx_train,:], axis=0)
    x1_train = np.append(x1_train, test1[idx_train,:], axis=0)
    x2_train = np.append(trunk2[idx_train,:], leaves2[idx_train,:], axis=0)
    x2_train = np.append(x2_train, branch2[idx_train,:], axis=0)
    x2_train = np.append(x2_train, test2[idx_train,:], axis=0)
    x3_train = np.append(trunk3[idx_train,:], leaves3[idx_train,:], axis=0)
    x3_train = np.append(x3_train, branch3[idx_train,:], axis=0)
    x3_train = np.append(x3_train, test3[idx_train,:], axis=0)
    x1_val = np.append(trunk1[idx_val,:], leaves1[idx_val,:], axis=0)
    x1_val = np.append(x1_val, branch1[idx_val,:], axis=0)
    x1_val = np.append(x1_val, test1[idx_val,:], axis=0)
    x2_val = np.append(trunk2[idx_val,:], leaves2[idx_val,:], axis=0)
    x2_val = np.append(x2_val, branch2[idx_val,:], axis=0)
    x2_val = np.append(x2_val, test2[idx_val,:], axis=0)
    x3_val = np.append(trunk3[idx_val,:], leaves3[idx_val,:], axis=0)
    x3_val = np.append(x3_val, branch3[idx_val,:], axis=0)
    x3_val = np.append(x3_val, test3[idx_val,:], axis=0)

    x1_train = np.reshape(x1_train, (-1, window_size, window_size, window_size, 1))
    x2_train = np.reshape(x2_train, (-1, window_size, window_size, window_size, 1))
    x3_train = np.reshape(x3_train, (-1, 1))
    x1_val = np.reshape(x1_val, (-1, window_size, window_size, window_size, 1))
    x2_val = np.reshape(x2_val, (-1, window_size, window_size, window_size, 1))
    x3_val = np.reshape(x3_val, (-1, 1))

    return x1_train, x2_train, x3_train, x1_val, x2_val, x3_val

trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3 = data_loader(what)

x1_train, x2_train, x3_train, x1_val, x2_val, x3_val = data_prep2(trunk1, trunk2, trunk3, leaves1, leaves2, leaves3, branch1, branch2, branch3, test1, test2, test3)

test_rands = np.random.permutation(trunk1.shape[0])[:test_size]
trunk1 = np.reshape(trunk1, (-1, window_size, window_size, window_size, 1))[test_rands,:]
trunk2 = np.reshape(trunk2, (-1, window_size, window_size, window_size, 1))[test_rands,:]
trunk3 = np.reshape(trunk3, (-1, 1))[test_rands,:]
leaves1 = np.reshape(leaves1, (-1, window_size, window_size, window_size, 1))[test_rands,:]
leaves2 = np.reshape(leaves2, (-1, window_size, window_size, window_size, 1))[test_rands,:]
leaves3 = np.reshape(leaves3, (-1, 1))[:test_size,:]
branch1 = np.reshape(branch1, (-1, window_size, window_size, window_size, 1))[test_rands,:]
branch2 = np.reshape(branch2, (-1, window_size, window_size, window_size, 1))[test_rands,:]
branch3 = np.reshape(branch3, (-1, 1))[:test_size,:]
test1 = np.reshape(test1, (-1, window_size, window_size, window_size, 1))[test_rands,:]
test2 = np.reshape(test2, (-1, window_size, window_size, window_size, 1))[test_rands,:]
test3 = np.reshape(test3, (-1, 1))[test_rands,:]

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

def supply_data():

    rands = np.random.randint(0, batch_size_tot-1, batch_size)

    x1 = x1_train[rands,:,:,:]
    x1 = np.reshape(x1, (batch_size, window_size, window_size, window_size, 1))
    x2 = x2_train[rands,:,:,:]
    x2 = np.reshape(x2, (batch_size, window_size, window_size, window_size, 1))

    labels = x3_train[rands]

    return x1, x2, labels

def supply_data2():

    # rands = np.random.randint(0, batch_size_tot-1, batch_size_train)

    # x1 = x1_train[rands,:,:,:]
    # x1 = np.reshape(x1, (batch_size_train, window_size, window_size, window_size, 1))
    # x2 = x2_train[rands,:,:,:]
    # x2 = np.reshape(x2, (batch_size_train, window_size, window_size, window_size, 1))

    # labels = x3_train[rands]

    # return x1, x2, labels

    rands = np.random.randint(0, int(batch_size_tot/4-1), int(batch_size_train/4))
    num = int(batch_size_tot/4)

    x1 = np.append(
        x1_train[rands,:,:,:],
        x1_train[num + rands,:,:,:],
        0)
    x1 = np.append(
        x1,
        x1_train[2*num + rands,:,:,:])
    x1 = np.append(
        x1,
        x1_train[3*num + rands,:,:,:])
    x1 = np.reshape(x1, (batch_size_train, window_size, window_size, window_size, 1))

    x2 = np.append(
        x2_train[rands,:,:,:],
        x2_train[num + rands,:,:,:],
        0)
    x2 = np.append(
        x2,
        x2_train[2*num + rands,:,:,:])
    x2 = np.append(
        x2,
        x2_train[3*num + rands,:,:,:])
    x2 = np.reshape(x2, (batch_size_train, window_size, window_size, window_size, 1))

    labels = np.append(
        x3_train[rands],
        x3_train[num + rands],
        0)
    labels = np.append(
        labels,
        x3_train[2*num + rands])
    labels = np.append(
        labels,
        x3_train[3*num + rands])
    labels = np.reshape(labels, (batch_size_train, 1))
    
    return x1, x2, labels

def stats(l2distance, margin, labels, mode):

    prd_match = tf.less_equal(l2distance - margin, 0.)
    prd_match = tf.reshape(prd_match, [batch_size_train, 1])

    prd_non_match = tf.greater(l2distance - margin, 0.)
    prd_non_match = tf.reshape(prd_non_match, [batch_size_train, 1])

    true_match = tf.equal(labels, 1.)
    true_match = tf.reshape(true_match, [batch_size_train, 1])

    true_non_match = tf.equal(labels, 0.)
    true_non_match = tf.reshape(true_non_match, [batch_size_train, 1])


    false_matches = tf.logical_and(prd_match, true_non_match)
    false_non_matches = tf.logical_and(prd_non_match, true_match)
    true_matches = tf.logical_and(prd_match, true_match)
    true_non_matches = tf.logical_and(prd_non_match, true_non_match)


    if mode == 'full':

        false_matches_sum = tf.reduce_sum(tf.cast(false_matches, tf.int32))
        false_non_matches_sum = tf.reduce_sum(tf.cast(false_non_matches, tf.int32))
        true_matches_sum = tf.reduce_sum(tf.cast(true_matches, tf.int32))
        true_non_matches_sum = tf.reduce_sum(tf.cast(true_non_matches, tf.int32))

        accuracy = tf.divide(
            (true_matches_sum + true_non_matches_sum),
            (true_matches_sum + true_non_matches_sum + false_matches_sum + false_non_matches_sum))

        error_rate = tf.divide(
            false_matches_sum,
            (false_matches_sum + true_non_matches_sum))

        recall = tf.divide(
            true_matches_sum,
            true_matches_sum + false_matches_sum)

        return 1.-accuracy, error_rate, recall, true_matches_sum, false_matches_sum, true_non_matches_sum, false_non_matches_sum

    if mode == 'split':

        false_matches_sum_trunk = tf.reduce_sum(tf.cast(false_matches[:int(batch_size_train/4)], tf.int32))
        false_non_matches_sum_trunk = tf.reduce_sum(tf.cast(false_non_matches[:int(batch_size_train/4)], tf.int32))
        true_matches_sum_trunk = tf.reduce_sum(tf.cast(true_matches[:int(batch_size_train/4)], tf.int32))
        true_non_matches_sum_trunk = tf.reduce_sum(tf.cast(true_non_matches[:int(batch_size_train/4)], tf.int32))

        accuracy_trunk =  tf.divide(
            (true_matches_sum_trunk + true_non_matches_sum_trunk),
            (true_matches_sum_trunk + true_non_matches_sum_trunk + false_matches_sum_trunk + false_non_matches_sum_trunk))

        error_rate_trunk =   tf.divide(
            false_matches_sum_trunk,
            (false_matches_sum_trunk + true_non_matches_sum_trunk))

        recall_trunk = tf.divide(
            true_matches_sum_trunk,
            true_matches_sum_trunk + false_matches_sum_trunk)

        tf.summary.scalar('false_matches_sum_trunk', false_matches_sum_trunk)
        tf.summary.scalar('false_non_matches_sum_trunk', false_non_matches_sum_trunk)
        tf.summary.scalar('true_matches_sum_trunk', true_matches_sum_trunk)
        tf.summary.scalar('true_non_matches_sum_trunk', true_non_matches_sum_trunk)


        false_matches_sum_branches = tf.reduce_sum(tf.cast(false_matches[int(batch_size_train/4+1):int(2*batch_size_train/4)], tf.int32))
        false_non_matches_sum_branches = tf.reduce_sum(tf.cast(false_non_matches[int(batch_size_train/4+1):int(2*batch_size_train/4)], tf.int32))
        true_matches_sum_branches = tf.reduce_sum(tf.cast(true_matches[int(batch_size_train/4+1):int(2*batch_size_train/4)], tf.int32))
        true_non_matches_sum_branches = tf.reduce_sum(tf.cast(true_non_matches[int(batch_size_train/4+1):int(2*batch_size_train/4)], tf.int32))

        accuracy_branches =  tf.divide(
            (true_matches_sum_branches + true_non_matches_sum_branches),
            (true_matches_sum_branches + true_non_matches_sum_branches + false_matches_sum_branches + false_non_matches_sum_branches))

        error_rate_branches =   tf.divide(
            false_matches_sum_branches,
            (false_matches_sum_branches + true_non_matches_sum_branches))

        recall_branches = tf.divide(
            true_matches_sum_branches,
            true_matches_sum_branches + false_matches_sum_branches)

        tf.summary.scalar('false_matches_sum_branches', false_matches_sum_branches)
        tf.summary.scalar('false_non_matches_sum_branches', false_non_matches_sum_branches)
        tf.summary.scalar('true_matches_sum_branches', true_matches_sum_branches)
        tf.summary.scalar('true_non_matches_sum_branches', true_non_matches_sum_branches)



        false_matches_sum_leaves = tf.reduce_sum(tf.cast(false_matches[int(2*batch_size_train/4+1):int(3*batch_size_train/4)], tf.int32))
        false_non_matches_sum_leaves = tf.reduce_sum(tf.cast(false_non_matches[int(2*batch_size_train/4+1):int(3*batch_size_train/4)], tf.int32))
        true_matches_sum_leaves = tf.reduce_sum(tf.cast(true_matches[int(2*batch_size_train/4+1):int(3*batch_size_train/4)], tf.int32))
        true_non_matches_sum_leaves = tf.reduce_sum(tf.cast(true_non_matches[int(2*batch_size_train/4+1):int(3*batch_size_train/4)], tf.int32))

        accuracy_leaves = tf.divide(
            (true_matches_sum_leaves + true_non_matches_sum_leaves),
            (true_matches_sum_leaves + true_non_matches_sum_leaves + false_matches_sum_leaves + false_non_matches_sum_leaves))

        error_rate_leaves = tf.divide(
            false_matches_sum_leaves,
            (false_matches_sum_leaves + true_non_matches_sum_leaves))

        recall_leaves = tf.divide(
            true_matches_sum_leaves,
            true_matches_sum_leaves + false_matches_sum_leaves)

        tf.summary.scalar('false_matches_sum_leaves', false_matches_sum_leaves)
        tf.summary.scalar('false_non_matches_sum_leaves', false_non_matches_sum_leaves)
        tf.summary.scalar('true_matches_sum_leaves', true_matches_sum_leaves)
        tf.summary.scalar('true_non_matches_sum_leaves', true_non_matches_sum_leaves)



        # false_matches_sum_test = tf.reduce_sum(tf.cast(false_matches[int(3*batch_size_train/4+1):int(4*batch_size_train/4)], tf.int32))
        # false_non_matches_sum_test = tf.reduce_sum(tf.cast(false_non_matches[int(3*batch_size_train/4+1):int(4*batch_size_train/4)], tf.int32))
        # true_matches_sum_test = tf.reduce_sum(tf.cast(true_matches[int(3*batch_size_train/4+1):int(4*batch_size_train/4)], tf.int32))
        # true_non_matches_sum_test = tf.reduce_sum(tf.cast(true_non_matches[int(3*batch_size_train/4+1):int(4*batch_size_train/4)], tf.int32))

        # accuracy_test =  tf.divide(
        #     (true_matches_sum_test + true_non_matches_sum_test),
        #     (true_matches_sum_test + true_non_matches_sum_test + false_matches_sum_test + false_non_matches_sum_test))

        # error_rate_test =   tf.divide(
        #     false_matches_sum_test,
        #     (false_matches_sum_test + true_non_matches_sum_test))

        return 1.-accuracy_trunk, 1.-accuracy_branches, 1.-accuracy_leaves, error_rate_trunk, error_rate_branches, error_rate_leaves, recall_trunk, recall_branches, recall_leaves

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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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
                # kernel_initializer=tf.random_normal_initializer(0.01, 1.),
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
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

        tf.summary.histogram('output_conv8', output)

        # images = tf.Variable(output, name='images')
        # saver = tf.train.Saver([images])
        

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
            nf =tf.constant(batch_size_train, tf.float32)

        with tf.name_scope('euclidean_distances'):

            l2distance = tf.sqrt(tf.reduce_mean(tf.pow(zero + tf.subtract(tf.reshape(output_1, [batch_size_train, -1]), tf.reshape(output_2, [batch_size_train, -1])),2), 1)) * tf.constant(1e+05)# + tf.constant(1.)

            l2distance_full_matrix = tf.sqrt(tf.pow(zero + tf.subtract(tf.reshape(output_1, [batch_size_train, -1]), tf.reshape(output_2, [batch_size_train, -1])), 2)) * tf.constant(1e+05)
            
            l2diff_match_vector = tf.reshape(labels, [1, -1]) * tf.reshape(l2distance, [1, -1])

            l2diff_non_match_vector = tf.reshape((1. - labels), [1, -1]) * tf.reshape(l2distance, [1, -1])
 
            ### sort trunk l2 distances ###
            values_matches_trunk = tf.contrib.framework.sort(
                l2diff_match_vector[:int(batch_size),:],
                axis=-1,
                direction='DESCENDING')[:batch_size]

            values_non_matches_trunk = tf.contrib.framework.sort(
                l2diff_non_match_vector[:int(batch_size),:],
                axis=-1,
                direction='ASCENDING')[:batch_size]

            ### sort branch l2 distances ###
            values_matches_branches = tf.contrib.framework.sort(
                l2diff_match_vector[int(batch_size_train/4):(int(batch_size_train/4) + int(batch_size)),:],
                axis=-1,
                direction='DESCENDING')[:batch_size]

            values_non_matches_branches = tf.contrib.framework.sort(
                l2diff_non_match_vector[int(batch_size_train/4):(int(batch_size_train/4) + int(batch_size)),:],
                axis=-1,
                direction='ASCENDING')[:batch_size]

            ### sort leaves l2 distances ###
            values_matches_leaves = tf.contrib.framework.sort(
                l2diff_match_vector[2*int(batch_size_train/4):(2*int(batch_size_train/4) + int(batch_size)),:],
                axis=-1,
                direction='DESCENDING')[:batch_size]

            values_non_matches_leaves = tf.contrib.framework.sort(
                l2diff_non_match_vector[2*int(batch_size_train/4):(2*int(batch_size_train/4) + int(batch_size)),:],
                axis=-1,
                direction='ASCENDING')[:batch_size]

            ### sort test l2 distances ###
            values_matches_test = tf.contrib.framework.sort(
                l2diff_match_vector[3*int(batch_size_train/4):(3*int(batch_size_train/4) + int(batch_size)),:],
                axis=-1,
                direction='DESCENDING')[:batch_size]

            values_non_matches_test = tf.contrib.framework.sort(
                l2diff_non_match_vector[3*int(batch_size_train/4):(3*int(batch_size_train/4) + int(batch_size)),:],
                axis=-1,
                direction='ASCENDING')[:batch_size]


            values_matches = tf.concat([
                values_matches_trunk,
                values_matches_branches,
                values_matches_leaves,
                values_matches_test],
                0)

            values_non_matches = tf.concat([
                values_non_matches_trunk,
                values_non_matches_branches,
                values_non_matches_leaves,
                values_non_matches_test],
                0)


            dummy_non_match = tf.cast(tf.less(values_non_matches - margin, 0.), tf.float32)

            values_non_matches_over_margin = values_non_matches * dummy_non_match

            l2diff_non_match1 = tf.reduce_sum(values_non_matches_over_margin) / (tf.reduce_sum(dummy_non_match) + zero)

            f1 = lambda: l2diff_non_match1
            f2 = lambda: tf.reduce_mean(values_non_matches)
            l2diff_non_match = tf.case([(tf.equal(l2diff_non_match1, 0.), f2)], default=f1)

            l2diff_match = tf.reduce_sum(values_matches)/(4*batch_size)

            # l2diff_non_match = tf.reduce_mean(values_non_matches)

            threshold = .5 * (l2diff_match + l2diff_non_match)

        with tf.name_scope('match_loss'):

            part1 = tf.pow(l2diff_match, 2)

        with tf.name_scope('non_match_loss'):
         
            partmax = tf.maximum(
                zero,
                tf.subtract(margin, l2diff_non_match))

            partmax = partmax * (tf.reduce_sum(dummy_non_match) + zero) / (4*batch_size)

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

            # prd_match = tf.less_equal(l2distance - margin, 0.)
            # prd_non_match = tf.greater(l2distance - margin, 0.)
            # true_match = tf.equal(labels, 1.)
            # true_non_match = tf.equal(labels, 0.)

            # false_matches = tf.logical_and(prd_match, true_non_match)
            # false_non_matches = tf.logical_and(prd_non_match, true_match)
            # true_matches = tf.logical_and(prd_match, true_match)
            # true_non_matches = tf.logical_and(prd_non_match, true_non_match)

            # false_matches_rate = tf.reduce_mean(tf.cast(false_matches, tf.float32))
            # false_non_matches_rate = tf.reduce_mean(tf.cast(false_non_matches, tf.float32))
            # false_matches_sum = tf.reduce_sum(tf.cast(false_matches, tf.int32))
            # false_non_matches_sum = tf.reduce_sum(tf.cast(false_non_matches, tf.int32))
            # true_matches_sum = tf.reduce_sum(tf.cast(false_matches, tf.int32))
            # true_non_matches_sum = tf.reduce_sum(tf.cast(false_non_matches, tf.int32))

            # accuracy =  tf.divide(
            #     (true_matches_sum + true_non_matches_sum),
            #     (true_matches_sum + true_non_matches_sum + false_matches_sum + false_non_matches_sum))

            # error_rate =   tf.divide(
            #     false_matches_sum,
            #     (false_matches_sum + true_non_matches_sum))


            accuracy, error_rate, recall, true_matches_sum, false_matches_sum, true_non_matches_sum, false_non_matches_sum = stats(l2distance, margin, labels, 'full')

            accuracy_trunk, accuracy_branches, accuracy_leaves, error_rate_trunk, error_rate_branches, error_rate_leaves, recall_trunk, recall_branches, recall_leaves = stats(l2distance, margin, labels, 'split')

            # accuracy_trunk_dyn, accuracy_branches_dyn, accuracy_leaves_dyn, accuracy_test_dyn, error_rate_trunk_dyn, error_rate_branches_dyn, error_rate_leaves_dyn, error_rate_test_dyn = stats(l2distance, threshold, labels, 'split')



        tf.summary.scalar('l2diff_match', tf.reshape(l2diff_match, []))
        tf.summary.scalar('l2diff_non_match', tf.reshape(l2diff_non_match, []))
        tf.summary.scalar('safety_margin', tf.subtract(l2diff_non_match, l2diff_match))
        tf.summary.scalar('match_loss', tf.reshape(part1, []))
        tf.summary.scalar('non_match_loss', tf.reshape(part2, []))
        tf.summary.scalar('loss', tf.reshape(loss, []))
        tf.summary.scalar('accuracy', tf.reshape(accuracy, []))
        tf.summary.scalar('error_rate', tf.reshape(error_rate, []))
        tf.summary.scalar('accuracy_trunk', tf.reshape(accuracy_trunk, []))
        tf.summary.scalar('accuracy_branches', tf.reshape(accuracy_branches, []))
        tf.summary.scalar('accuracy_leaves', tf.reshape(accuracy_leaves, []))
        tf.summary.scalar('error_rate_trunk', tf.reshape(error_rate_trunk, []))
        tf.summary.scalar('error_rate_branches', tf.reshape(error_rate_branches, []))
        tf.summary.scalar('error_rate_leaves', tf.reshape(error_rate_leaves, []))
        # tf.summary.scalar('error_rate_trunk_dyn', error_rate_trunk_dyn)
        # tf.summary.scalar('error_rate_branches_dyn', error_rate_branches_dyn)
        # tf.summary.scalar('error_rate_leaves_dyn', error_rate_leaves_dyn)
        # tf.summary.scalar('error_rate_test_dyn', error_rate_trunk_dyn)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('recall_trunk', recall_trunk)
        tf.summary.scalar('recall_branches', recall_branches)
        tf.summary.scalar('recall_leaves', recall_leaves)
        tf.summary.histogram('distances', l2distance)
        tf.summary.scalar('threshold', tf.reshape(threshold, []))     
        tf.summary.scalar('true_matches_sum', tf.reshape(true_matches_sum, []))     
        tf.summary.scalar('false_matches_sum', tf.reshape(false_matches_sum, []))     
        tf.summary.scalar('true_non_matches_sum', tf.reshape(true_non_matches_sum, []))     
        tf.summary.scalar('false_non_matches_sum', tf.reshape(false_non_matches_sum, []))     


        return loss, l2diff_match, l2diff_non_match, threshold, l2diff_match_vector, l2diff_non_match_vector, l2distance_full_matrix, error_rate, error_rate_trunk, error_rate_branches, error_rate_leaves

def dist(output1, output2):

    with tf.name_scope('distance_comp'):

        distances =  tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(tf.reshape(output1, [test_size, -1]), tf.reshape(output2, [test_size, -1])),2), 1)) * tf.constant(1e+05)

        return tf.reshape(distances, [-1])

def online_data_supply(x1_train, x2_train, x3_train, l2diff_match_vector, l2diff_non_match_vector, batch_size, i):


    if i == 0:

        return x1_train[:2*batch_size, :], x2_train[:2*batch_size, :], x3_train[:2*batch_size, :]

    else:

        values_matches, indices_matches = tf.nn.top_k(
            l2diff_match_vector,
            k=batch_size,
            sorted=True)

        print(indices_matches)

        values_non_matches, indices_non_matches = tf.nn.top_k(
            l2diff_non_match_vector,
            k=batch_size,
            sorted=True)

        x1online = tf.concat(
            [x1_train[indices_matches,:],
                    x1_train[indices_non_matches,:]],
            0)

        x2online = tf.concat(
            [x2_train[indices_matches,:],
                    x2_train[indices_non_matches,:]],
            0)

        x3online = tf.concat(
            [x3_train[indices_matches,:],
                    x3_train[indices_non_matches,:]],
            0)


        return x1online, x2online, x3online

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


# embedding_var = tf.Variable(tf.truncated_normal([100, 10]), name='embedding')


x1 = tf.placeholder(
    tf.float32, 
    [batch_size_train, window_size, window_size, window_size, 1],
    name="input1")

x2 = tf.placeholder(
    tf.float32, 
    [batch_size_train, window_size, window_size, window_size, 1],
    name="input2")

labels = tf.placeholder(
    tf.float32, 
    [batch_size_train, 1],
    name="input3")

# x1_test = tf.placeholder(
#     tf.float32, 
#     [test_size, window_size, window_size, window_size, 1],
#     name="input1_test")

# x2_test = tf.placeholder(
#     tf.float32, 
#     [test_size, window_size, window_size, window_size, 1],
#     name="input2_test")

# labels_test = tf.placeholder(
#     tf.float32, 
#     [test_size, 1],
#     name="input3_test")


output1 = siamese(x1, reuse=False)
output2 = siamese(x2, reuse=True)
# output1_test = siamese(x1_test, reuse=True)
# output2_test = siamese(x2_test, reuse=True)

# test_distances = dist(output1, output2)


loss, l2diff_match, l2diff_non_match, threshold, l2diff_match_vector, l2diff_non_match_vector, l2distance_full_matrix, error_rate, error_rate_trunk, error_rate_branches, error_rate_leaves = contrastive_loss(output1, output2, labels)


# loss_val, l2diff_match_val, l2diff_non_match_val, threshold_val, l2diff_match_vector_val, l2diff_non_match_vector_val, l2distance_full_matrix_val, error_rate_val, error_rate_trunk_val, error_rate_branches_val, error_rate_leaves_val = contrastive_loss(output1_test, output2_test, labels_test)
# embedding_var = tf.Variable(tf.zeros(l2distance_full_matrix.shape), name='embedding')

# embedding_var.assign(l2distance_full_matrix)


# config = projector.ProjectorConfig()

# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name


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


    saver = tf.train.Saver()


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_
        train_step = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.99,
            beta2=0.999,
            epsilon=1e-08,
            use_locking=False,
            name='Adam'
            ).minimize(loss)

    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(
        DATA_DIR,
        graph = sess.graph)




    # config = projector.ProjectorConfig()

    # embedding = config.embeddings.add()
    # embedding.tensor_name = embedding_var.name


    for i in range(0,iterations):


        data1, data2, data3 = supply_data2()


        # metadata = os.path.join(DATA_DIR, 'metadata.tsv')

        # with open(metadata, 'w') as metadata_file:
        #     for row in data3:
        #         metadata_file.write('%d\n' % row)



        summary, _ , loss_train, l2md, l2nonmd, thresh, error_rate_train, error_rate_trunk_train, error_rate_branches_train, error_rate_leaves_train = sess.run([
            merged, 
            train_step, 
            loss, 
            l2diff_match, 
            l2diff_non_match, 
            threshold, 
            error_rate, 
            error_rate_trunk, 
            error_rate_branches, 
            error_rate_leaves
            ], 
            feed_dict={
                x1: data1,
                x2: data2,
                labels: data3
                })


        # ### create the tensorboard embedding ###

        # sess.run(embedding_var.initializer)

        # embedding_var.assign(l2distance_full_matrix)

        # # sess.run(embedding_var.initializer)
        # saver.save(sess, os.path.join(DATA_DIR, 'embedding_var.ckpt'))

        # config = projector.ProjectorConfig()
        # # One can add multiple embeddings.
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        # embedding.metadata_path = 'metadata.tsv'
        # # Link this tensor to its metadata file (e.g. labels).
        # # embedding.metadata_path = metadata
        # # Saves a config file that TensorBoard will read during startup.
        # projector.visualize_embeddings(tf.summary.FileWriter(DATA_DIR), config)

        # projector.visualize_embeddings(tf.summary.FileWriter(DATA_DIR), config)

        # saver.save(sess, os.path.join(DATA_DIR, 'filename.ckpt'))



        threshold_history[i] = thresh

        test_threshold = moving_threshold(threshold_history, i)


        train_writer.add_summary(summary, i)
        saver.save(sess, os.path.join(DATA_DIR, "model.ckpt"), i)


        if i % 1 == 0:

            print('########################################################################### iteration ', i)
            print('@', DATA_DIR, 'margin->', L2margin, ' | batch size->', batch_size_train)
            print('worst', batch_size*4, '(', batch_size, 'per category) samples used to train')
            print('###########################################################################')
            print('loss...........................................', loss_train)
            print('l2diff_match...................................', l2md)
            print('l2diff_non_match...............................', l2nonmd)
            print('error_rate_train...............................', error_rate_train)
            print('error_rate_trunk_train.........................', error_rate_trunk_train)
            print('error_rate_branches_train......................', error_rate_branches_train)
            print('error_rate_leaves_train........................', error_rate_leaves_train)
            # print(test_dist)
            # print(trunk3)

        if i % acc_print_freq == 0 and i > 0:

            # full_matrix_trunks, error_rate_train_trunks = sess.run([l2distance_full_matrix_val, error_rate_val], feed_dict={
            #     x1_test: trunk1,
            #     x2_test: trunk2,
            #     labels_test: trunk3})

            # print('error_rate --> trunks..................', error_rate_train_trunks)

            # filename_dist_trunks = DATA_DIR + 'distances_trunks_' + str(i) + '.npy'
            # filename_labels_trunks = DATA_DIR + 'labels_trunks_' + str(i) + '.npy'
            # print('saving trunk distances/ labels to ', filename_dist_trunks, filename_labels_trunks)
            # np.save(filename_dist_trunks, full_matrix_trunks)
            # if i == 0:
            #     np.save(filename_labels_trunks, trunk3)


            # trunk_acc =  monitor_performance(test_dist_trunk, trunk3, L2margin)
            # trunk_acc_t =  monitor_performance(test_dist_trunk, trunk3, test_threshold)
            # print('trunk performance---------------------------------------', trunk_acc)
            # print('trunk performance dynamic threshold---------------------', trunk_acc_t)


            # test_dist_branch = sess.run(test_distances, feed_dict={
            #     x1_test: branch1,
            #     x2_test: branch2})

            # branch_acc =  monitor_performance(test_dist_branch, branch3, L2margin)
            # branch_acc_t =  monitor_performance(test_dist_branch, branch3, test_threshold)
            # print('branch performance---------------------------------------', branch_acc)
            # print('branch performance dynamic threshold---------------------', branch_acc_t)


            # test_dist_leaves = sess.run(test_distances, feed_dict={
            #     x1_test: leaves1,
            #     x2_test: leaves2})

            # leaves_acc =  monitor_performance(test_dist_leaves, leaves3, L2margin)
            # leaves_acc_t =  monitor_performance(test_dist_leaves, leaves3, test_threshold)
            # print('leaves performance---------------------------------------', leaves_acc)
            # print('leaves performance dynamic threshold---------------------', leaves_acc_t)

            # test_dist = sess.run(test_distances, feed_dict={
            #     x1_test: test1,
            #     x2_test: test2})

            # test_acc =  monitor_performance(test_dist, test3, L2margin)
            # test_acc_t =  monitor_performance(test_dist, test3, test_threshold)
            # print('test performance---------------------------------------', test_acc)
            # print('test performance dynamic threshold---------------------', test_acc_t)

            for h in range(test_multiplier):

                if h % 2 == 0:

                    progress = h/test_multiplier*100
                    progress = np.round(progress)

                    print(progress, '%', 'of sampling done.')

                if h == 0:

                    storage_distances = []
                    storage_labels = []

                data1_test, data2_test, data3_test = supply_data2()
                
                full_matrix_test = sess.run(l2distance_full_matrix, feed_dict={
                x1: data1_test,
                x2: data2_test})

                storage_distances.append(full_matrix_test)
                storage_labels.append(data3_test)

            storage_distances = np.asarray(storage_distances)
            storage_labels = np.asarray(storage_labels)

            filename_dist = DATA_DIR + 'distances_' + str(i) + '.npy'
            filename_labels = DATA_DIR + 'labels_' + str(i) + '.npy'
            print('saving distances to ', filename_dist)
            np.save(filename_dist, storage_distances)
            np.save(filename_labels, storage_labels)


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


        if np.isnan(loss_train):
            print('Model diverged with loss = NaN')
            quit()


# print(w_1)