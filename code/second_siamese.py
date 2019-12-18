from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import csv
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.preprocessing import maxabs_scale
from builtins import input

# tf.logging.set_verbosity(tf.logging.INFO)
DATA_DIR = '/Users/benediktdietz/deep_tf/train/'

print('======================================================================')
print('======================================================================')
print('=========================== brain dementia ===========================')
print('======================================================================')
print('======================================================================')
print('saving graph to ', DATA_DIR)
print('import training brains................................................')
x_train = np.load('X_train.npy')[1:,:]
# x_train = np.asarray(x_train, dtype=np.float64)
# x_train = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(x_train)
x_train = x_train.reshape((-1, 176, 208, 176, 1))
print('import training labels................................................')
y_train = pd.read_csv('y_1.csv')
y_train = np.asarray(y_train)

labels = np.zeros(y_train.shape[0])

for i in range(0,len(y_train)):
    
    if y_train[i] >= 50:
        
        labels[i] += 1

y_train = labels

print(y_train)


print(np.sum(labels))

batch_size = x_train.shape[0]


print('======================================================================')
# print('size of 3d brain pixels..............................', x_train3d.shape)
print('size of training brains..............................', x_train.shape)
print('size of training labels..............................', y_train.shape)


def flatten(x):

        x_flat = tf.reshape(x, [batch_size, -1])

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

def suppy_data(x_train, y_train, numbrains):

        random1 = np.random.randint(0, batch_size, numbrains)
        random2 = np.random.randint(0, batch_size, numbrains)
        print('randoms------', random1, random2)
        x1 = x_train[random1,:,:,:,:]
        x2 = x_train[random2,:,:,:,:]

        y = []

        if y_train[random1] == y_train[random2]:
            y = 0
        elif y_train[random1] != y_train[random2]:
            y = 1

        y = (y_train[random1] == y_train[random2]).astype('float')

        print('yyyyyyyyyy', y)

        return x1, x2, y




print('======================================================================')
print('=========================== building graph ===========================')
print('======================================================================')



class siamese():


        def __init__(self):

                self.x1 = tf.placeholder(
                        tf.float32, 
                        [1, 176, 208, 176, 1],
                        name="input1")

                self.x2 = tf.placeholder(
                        tf.float32, 
                        [1, 176, 208, 176, 1],
                        name="input1")

                # with tf.variable_scope("siamese", reuse=True) as scope:
                #     self.o1 = self.network(self.x1)
                #     scope.reuse_variables()
                #     self.o2 = self.network(self.x2)

                with tf.variable_scope("siamese") as scope:
                    self.o1 = self.network(self.x1)
                    scope.reuse_variables()
                    self.o2 = self.network(self.x2)

                self.y = tf.placeholder(
                        tf.float32,
                        [1, ],
                        name="labels")

                self.loss = self.contrastive_loss()



        def conv_relu(self, input, kernel_shape, bias_shape):

            # Create variable named "weights".
            weights = tf.get_variable(
                "weights", 
                kernel_shape,
                initializer=tf.random_normal_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            # Create variable named "biases".
            biases = tf.get_variable("biases", bias_shape,
                initializer=tf.constant_initializer(0.0),
                regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            conv = tf.nn.conv3d(
                input,
                weights,
                (1, 1, 1, 1, 1),
                padding='VALID',
                data_format='NDHWC',
                name='conv'
                )


            return tf.nn.relu(conv + biases)


        def network(self, x):


            with tf.variable_scope("conv_relu1"):

            
                # mean, variance = tf.nn.moments(
                #     x_norm,
                #     axes=[0, 1, 2, 3],
                #     shift=None,
                #     name='moments',
                #     keep_dims=False
                #     )

                x_norm = tf.nn.batch_normalization(
                    x,
                    mean=0,
                    variance=1,
                    offset=None,
                    scale=None,
                    variance_epsilon=1e-06,
                    name=None
                    )

                pool1 = tf.nn.max_pool3d(
                    x_norm,
                    [1, 4, 4, 4, 1],
                    (1, 4, 4, 4, 1),
                    padding='VALID',
                    data_format='NDHWC',
                    name='pool1'
                    )

                tf.summary.histogram(
                    'pool1', 
                    pool1)

                print('shape of pool1', pool1.shape)

                conv_relu1 = self.conv_relu(
                    pool1, 
                    [3, 3, 3, 1, 16], 
                    [16])

                conv_relu1_norm = tf.nn.batch_normalization(
                    conv_relu1,
                    mean=0,
                    variance=1,
                    offset=None,
                    scale=None,
                    variance_epsilon=1e-06,
                    name=None
                    )

                tf.summary.histogram(
                    'conv_relu1_norm', 
                    conv_relu1_norm)

                print('shape of conv4', conv_relu1.shape)

            with tf.variable_scope("conv_relu2_pool"):

                conv_relu2 = self.conv_relu(
                    conv_relu1_norm, 
                    [3, 3, 3, 16, 16],
                    [16])

                conv_relu2_norm = tf.nn.batch_normalization(
                    conv_relu2,
                    mean=0,
                    variance=1,
                    offset=None,
                    scale=None,
                    variance_epsilon=1e-06,
                    name=None
                    )

                tf.summary.histogram(
                    'conv_relu2_norm', 
                    conv_relu2_norm)

                # conv_relu2_pool = tf.layers.max_pooling3d(
                #     inputs = conv_relu2,
                #     pool_size=[3, 3, 3],
                #     strides=(2, 2, 2),
                #     padding='valid',
                #     name='pool1'
                #     )

                conv_relu2_pool = tf.nn.max_pool3d(
                    conv_relu2_norm,
                    [1, 3, 3, 3, 1],
                    (1, 2, 2, 2, 1),
                    padding='VALID',
                    data_format='NDHWC',
                    name='conv_relu2_pool'
                    )

                tf.summary.histogram(
                    'conv_relu2_pool', 
                    conv_relu2_pool)

                print('shape of conv4', conv_relu2_pool.shape)

            with tf.variable_scope("conv_relu3"):

                conv_relu3 = self.conv_relu(
                    conv_relu2_pool, 
                    [3, 3, 3, 16, 32],
                    [32])

                conv_relu3_norm = tf.nn.batch_normalization(
                    conv_relu3,
                    mean=0,
                    variance=1,
                    offset=None,
                    scale=None,
                    variance_epsilon=1e-06,
                    name=None
                    )

                tf.summary.histogram(
                    'conv_relu3_norm', 
                    conv_relu3_norm)

                print('shape of conv4', conv_relu3.shape)

            with tf.variable_scope("conv_relu4"):

                conv_relu4 = self.conv_relu(
                    conv_relu3, 
                    [3, 3, 3, 32, 32],
                    [32])

                conv_relu4_norm = tf.nn.batch_normalization(
                    conv_relu4,
                    mean=1e-06,
                    variance=1,
                    offset=None,
                    scale=None,
                    variance_epsilon=1e-06,
                    name=None
                    )

                tf.summary.histogram(
                    'conv_relu4_norm', 
                    conv_relu4_norm)

                print('shape of conv4', conv_relu4.shape)

                return conv_relu4_norm



        def contrastive_loss(self):

            with tf.name_scope("contrastive_loss"):

                with tf.name_scope('constants'):

                    margin = tf.constant(1.)  
                    one = tf.constant(1.)  
                    zero = tf.constant(1e-06) 

                # l2diff = tf.losses.mean_squared_error(
                #     tf.reshape(self.o1, [-1]),
                #     tf.reshape(self.o2, [-1]),
                #     weights=1e-03,
                #     scope=None,
                #     loss_collection=tf.GraphKeys.LOSSES,
                #     )

                with tf.name_scope('euclidean_distance'):

                    l2diff = tf.sqrt(
                        tf.reduce_mean(
                            tf.square(
                                tf.subtract(
                                    tf.reshape(self.o1, [-1]),
                                    tf.reshape(self.o2, [-1])))))

                    l2diff = tf.multiply(
                        l2diff,
                        tf.constant(1e-03))

                with tf.name_scope('labels'):

                    labels = tf.to_float(self.y)

                with tf.name_scope('match_loss'):

                    part1 = tf.multiply(
                        tf.subtract(
                            one,
                            labels),
                        tf.square(
                            l2diff))

                with tf.name_scope('non_match_loss'):
                
                    partmax = tf.maximum(
                        zero,
                        tf.subtract(margin, l2diff))

                    part2 = tf.multiply(
                        labels,
                        np.square(
                            partmax))

                with tf.name_scope('regularization'):

                    reg_losses = tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES)

                    reg_constant = 0.001 
                    regularization = reg_constant * sum(reg_losses)

                with tf.name_scope('final_loss'):

                    loss_non_reg = np.add(part1, part2)
                    loss = np.add(loss_non_reg, regularization)


                tf.summary.scalar('l2diff', tf.reshape(l2diff, []))
                tf.summary.scalar('match_loss', tf.reshape(part1, []))
                tf.summary.scalar('non_match_loss', tf.reshape(part2, []))
                tf.summary.scalar('regularization', tf.reshape(regularization, []))
                tf.summary.scalar('loss', tf.reshape(loss, []))


                return loss



sess = tf.InteractiveSession()

siamese = siamese()

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(siamese.loss)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(
                DATA_DIR,
                graph = sess.graph)

# saver = tf.train.Saver()
tf.global_variables_initializer().run()

for i in range(0,5000):

    x1, x2, y = suppy_data(x_train, y_train, 1)
    print(x1.shape)
    print(x2.shape)
    print(y.shape)

    summary, _ , loss_train = sess.run([merged, train_step, siamese.loss], feed_dict={
        siamese.x1: x1,
        siamese.x2: x2,
        siamese.y: y})

    train_writer.add_summary(summary, i)


    if np.isnan(loss_train):
        print('Model diverged with loss = NaN')
        quit()






