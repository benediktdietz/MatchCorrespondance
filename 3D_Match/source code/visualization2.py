import argparse
import os
import shutil

from numpy import array
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from loadTrainDataGPU2 import data_loader
from loadTrainDataGPU2 import datalist_loader
from loadDADataGPU import targetlist_loader
from loadDADataGPU import sourcelist_loader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Match3DNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Match3DNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(64),
            
            # conv2
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(64),
            
            # maxpool
            nn.MaxPool3d(kernel_size=2, stride=2),
            # conv3
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(128),
            
            # conv4
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(128),
            
            # conv5
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(256),
            
            # conv6
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(256),
            
            # conv7
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(512),
            
            # conv8
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(512),
            nn.ReLU(inplace=True), 
            #nn.BatchNorm3d(512),       
            )

    def forward_once(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.

        forward one type of the input
        """
        x = self.features(x)
        x = x.view(x.size(0), 512) # reshpe it into (batch_size, feature_dimention)
        return x

    def forward(self, p1, p2, p3):
        """
        forward three times to compute the features of matched pair (p1,p2) and nonmatch pair (p1,p3)
        """
        p1 = self.forward_once(p1)
        p2 = self.forward_once(p2)
        p3 = self.forward_once(p3)
        return p1, p2, p3

def main():

    # create model
    batch_size = 200
    weight = 'checkpoint_5200.pth'
    datapath = '/home/drzadmin/Desktop/3DMatch-pytorch'
    testdir_indoor = datapath + '/test/indoor_test/test'
    testdir_tree = datapath + '/test/tree_test/test'
    testdir_trunk = datapath + '/test/tree_test/trunk_test'
    testdir_branch = datapath + '/test/tree_test/branch_test'
    testdir_leaves = datapath + '/test/tree_test/leaves_test'

    model = Match3DNet()

    print("=> loading checkpoint '{}'".format(weight))
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (iter {})".format(weight, checkpoint['iter']))

    # Data loading code

    model.eval()

    # load test data
    f = h5py.File(testdir_indoor + '.mat')
    data = f['data']
    labels = f['labels']

    descDistsall = []
    labels_all = []

    for i in range(batch_size):
        label = f[labels[0,i]][0][0]

        data_p1 = torch.FloatTensor(f[f['data'][0,i]]['voxelGridTDF'][:].reshape(1,1,30,30,30))
        data_p2 = torch.FloatTensor(f[f['data'][1,i]]['voxelGridTDF'][:].reshape(1,1,30,30,30))

        data_p1 = torch.autograd.Variable(data_p1)
        data_p2 = torch.autograd.Variable(data_p2)

        # compute output
        DesP1, DesP2, _ = model(data_p1, data_p2, data_p2)

        DesP1, DesP2 = DesP1.data.cpu().numpy(), DesP2.data.cpu().numpy()

        descDistsall.append(DesP1)
        descDistsall.append(DesP2)
        labels_all.append(label)
    
        # measure accuracy and record loss

    # load test data
    f = h5py.File(testdir_tree + '.mat')
    data = f['data']
    labels = f['labels']

    for i in range(batch_size):
        label = f[labels[0,i]][0][0]

        data_p1 = torch.FloatTensor(f[f['data'][0,i]]['voxelGridTDF'][:].reshape(1,1,30,30,30))
        data_p2 = torch.FloatTensor(f[f['data'][1,i]]['voxelGridTDF'][:].reshape(1,1,30,30,30))

        data_p1 = torch.autograd.Variable(data_p1)
        data_p2 = torch.autograd.Variable(data_p2)

        # compute output
        DesP1, DesP2, _ = model(data_p1, data_p2, data_p2)

        DesP1, DesP2 = DesP1.data.cpu().numpy(), DesP2.data.cpu().numpy()
        
        descDistsall.append(DesP1)
        descDistsall.append(DesP2)
        labels_all.append(label)

    X = TSNE(n_components=2).fit_transform(array(descDistsall).reshape(batch_size*4,512))
    X1 = X[:batch_size,:]
    X2 = X[batch_size:,:]

    points1 = plt.plot(X1[:,0], X1[:,1], 'ro', label='indoor samples')
    points2 = plt.plot(X2[:,0], X2[:,1], 'bo', label='tree samples')
    
    plt.legend()

    plt.show()
    plt.savefig('pretain_infoor_treesamples.png')
    
if __name__ == '__main__':
    main()