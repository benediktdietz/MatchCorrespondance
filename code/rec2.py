from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from tensorflow.python.util.tf_export import tf_export
import os
import glob
from numpy import array
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial import cKDTree
import math
from numpy.linalg import norm
from numpy import dot
import time
from matplotlib import pyplot as plt
import re
import sys
from struct import *
import imageio
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import Delaunay
from mpl_toolkits import mplot3d
import argparse
import tensorflow as tf
import csv
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.preprocessing import maxabs_scale, StandardScaler
import torch
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug
from sympy import nsolve, Eq, Symbol, sin, cos
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, RANSACRegressor

## Network Setting
#########################
DATA_DIR = 'tensorflow_logs/doublette_test5/'

iterations = 1000000 # range for training loop
batch_size = 8 # determines partition of loaded TDFs to be used for backpropagation
batch_size_train = 16 # batch size. input size = [batch_size_train, 3, window_size, window_size, window_size, 1]

learning_rate = 0.0005 # used in defined optimizer

L2margin = 100. # for contrastive loss computation

rec_iteration = 2000 # frequency of sampling for reconstruction test
rec_num = 1000 # range of loop through point combinations

acc_print_freq = 250 # frequency of sampling on validation set
out_freq = 50 # terminal output frequency

zero = tf.constant(1e-04)
#########################

## Data Loader Settings --> Mining from pictures
#########################
DataPath = 'TreeDataJake/' # path to tree images and resp. data
trees = np.asarray(['Cherry', 'KoreanStewartia'])
num_tree_folders = 100
num_tree_pics = np.asarray([789, 789, 709, 429])

voxelGridPatchRadius = 15  # in voxels
voxelSize = 0.01  # in meters
voxelMargin = voxelSize * 5

## Data Loader Settings --> Use preloaded TDFs (recommended for speed)
#########################
path = 'pre_loaded_TDFs/'
numfiles = 8 # number of file packs used for training loop
#########################

## Other Settings/ Debugging
#########################
window_size = 2 * voxelGridPatchRadius

random_tree_debug = 48
snapshot1_index_debug = 3
rand_row_debug, rand_col_debug = 117, 252
snapshot2_index_debug = 2

def dist_calc(coords, other_coords):
    
    xdistance = np.power(coords[0] - other_coords[0], 2)
    ydistance = np.power(coords[1] - other_coords[1], 2)
    zdistance = np.power(coords[2] - other_coords[2], 2)
    
    dist = np.sqrt(xdistance + ydistance + zdistance)
    
    return dist

def convert_pfm(file, debug=False):

    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type=f.readline().decode('latin-1')
        if "PF" in type:
           channels=3
        elif "Pf" in type:
           channels=1
        else:
           print("ERROR: Not a valid PFM file",file=sys.stderr)
           sys.exit(1)
        if(debug):
           print("DEBUG: channels={0}".format(channels))

        # Line 2: width height
        line=f.readline().decode('latin-1')
        width,height=re.findall('\d+',line)
        width=int(width)
        height=int(height)
        if(debug):
           print("DEBUG: width={0}, height={1}".format(width,height))

        # Line 3: +ve number means big endian, negative means little endian
        line=f.readline().decode('latin-1')
        BigEndian=True
        if "-" in line:
           BigEndian=False
        if(debug):
           print("DEBUG: BigEndian={0}".format(BigEndian))

        # Slurp all binary data
        samples = width*height*channels;
        buffer  = f.read(samples*4)

        # Unpack floats with appropriate endianness
        if BigEndian:
           fmt=">"
        else:
           fmt="<"
        fmt= fmt + str(samples) + "f"

        img = unpack(fmt,buffer)
        log_img = np.log(img)

        
        return np.reshape(np.asarray(img), (424, 512))

def file_index(snap_index):

    if snap_index < 10:
        file_index = '0000' + str(snap_index)
    elif snap_index < 100:
        file_index = '000' + str(snap_index)
    elif snap_index >= 100:
        file_index = '00' + str(snap_index)

    return str(file_index)

def get_cam_intrinsics():
    
    fx = 365.605889726
    fy = 365.605889726
    cx = 255
    cy = 211

    return fx, fy, cx, cy

def get_cam_position(img_number, random_tree_path):

    data_table = np.asarray(pd.read_table(random_tree_path + '/poses.txt', sep='\s', header=0, index_col=False, engine='python'))
    # print(pd.read_table('TreeData/poses.txt', sep='\s', header=0, index_col=False, engine='python'))
    index = np.squeeze(np.where(data_table[:,0] == img_number))
    #index = index[0]
    index_data = data_table[index]

    return index_data[1:]

def angle_between(a,b):
  arccosInput = dot(a,b)/norm(a)/norm(b)
  arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
  arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
  return math.acos(arccosInput)

def build_rot_mat(angle, axis):

    s = np.sin(angle)
    c = np.cos(angle)

    if axis == 'x':
        rot_mat = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    
    if axis == 'y':
        rot_mat = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    
    if axis == 'z':
        rot_mat = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    # print()
    # print('computed rotation matrix for ' + str(axis) + '-axis, angle: ', np.round(angle, 2), ':')
    # print(np.round(rot_mat, 2))
    # print()
    
    return np.asarray(rot_mat, np.float32)

def bbox_corners(ptCam):
    # Compute bounding box in pixel coordinates
    bboxRange = np.array([[ptCam[0]-voxelGridPatchRadius*voxelSize, ptCam[0]+voxelGridPatchRadius*voxelSize], 
            [ptCam[1]-voxelGridPatchRadius*voxelSize, ptCam[1]+voxelGridPatchRadius*voxelSize],
            [ptCam[2]-voxelGridPatchRadius*voxelSize, ptCam[2]+voxelGridPatchRadius*voxelSize]])
    
    bboxCorners = np.array([[bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1]],
            [bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1],bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1]],
            [bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1]]])
    bboxRange = np.reshape(bboxRange,(3,2))
    bboxCorners = np.reshape(bboxCorners,(3,8))

    return bboxCorners

def getPatchData(pointData,voxelGridPatchRadius,voxelSize,voxelMargin):

    output = False
    plot = False

    fx, fy, cx, cy = get_cam_intrinsics()

    depthIm = pointData['depthIm']
    depthIm[depthIm > 8] = 0


    # lowBoundX = int(np.min(pointData['bboxRangePixels'][1,:]))
    # highBoundX = int(np.max(pointData['bboxRangePixels'][1,:]))+1
    # lowBoundY = int(np.min(pointData['bboxRangePixels'][0,:]))
    # highBoundY = int(np.max(pointData['bboxRangePixels'][0,:]))+1


    lowBoundX = int(np.min(pointData['bboxRangePixels'][1,:]))
    highBoundX = int(np.max(pointData['bboxRangePixels'][1,:]))+1
    lowBoundY = int(np.min(pointData['bboxRangePixels'][0,:]))
    highBoundY = int(np.max(pointData['bboxRangePixels'][0,:]))+1

    depthPatch = depthIm[lowBoundX-1:highBoundX-1, lowBoundY-1:highBoundY-1]


    # Get TDF voxel grid local patches
    [pixX,pixY] = np.mgrid[lowBoundY:highBoundY, lowBoundX:highBoundX]
    pixX, pixY = pixX.T, pixY.T


    if pixX.shape != depthPatch.shape or pixY.shape != depthPatch.shape:
        if output:
            print()
            print()
            print('###################################')
            print('###### fatal dimension error ######')
            print('###################################')
            print()
            print()

        error_solution = np.asarray(np.zeros((1, window_size, window_size, window_size)))

        return error_solution

    else:


        camX = np.array((pixX-cx)*depthPatch/fx)
        camY = np.array(-(pixY-cy)*depthPatch/fy)
        camZ = np.array(depthPatch)

        

        # ValidX,ValidY = np.nonzero(depthPatch)[0], np.nonzero(depthPatch)[1]
        # ValidDepth = ValidX*np.shape(depthPatch)[1] + ValidY
        

        ValidX,ValidY = np.array(np.nonzero(depthPatch))
        ValidDepth = ValidX*np.shape(depthPatch)[1] + ValidY
        

        camPts = np.append(np.reshape(camX.T,(np.size(camX),1)),np.reshape(camY.T,(np.size(camY),1)),1)
        camPts = np.append(camPts,np.reshape(camZ.T,(np.size(camZ),1)),1)
        #camPts = np.array([camX,camY,camZ])
        #camPts = np.reshape(camPts,(np.size(depthPatch),3))
        # camPts = camPts[ValidDepth,:]



        #gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[(pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize, 
        #   np.float16(pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2):np.float16(pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize,
        #   (pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize]

        lowBoundX = pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2
        highBoundX = pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
        lowBoundY = pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2
        highBoundY = pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
        lowBoundZ = pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2
        highBoundZ = pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1



        gridPtsCamX, gridPtsCamY, gridPtsCamZ = np.mgrid[lowBoundX:highBoundX:voxelSize, lowBoundY:highBoundY:voxelSize, lowBoundZ:highBoundZ:voxelSize]

        gridPtsCam = np.append(np.reshape(gridPtsCamX.T,(np.size(gridPtsCamX),1)),np.reshape(gridPtsCamY.T,(np.size(gridPtsCamX),1)),1)
        gridPtsCam = np.append(gridPtsCam,np.reshape(gridPtsCamZ.T,(np.size(gridPtsCamZ),1)),1)


        # n1 = torch.FloatTensor(camPts).cuda()


        # part = int(np.round(gridPtsCam.shape[0]/5))

        # n2_1 = torch.FloatTensor(gridPtsCam[:part,:]).cuda()
        # n2_2 = torch.FloatTensor(gridPtsCam[part:2*part,:]).cuda()
        # n2_3 = torch.FloatTensor(gridPtsCam[2*part:3*part,:]).cuda()
        # n2_4 = torch.FloatTensor(gridPtsCam[3*part:4*part,:]).cuda()
        # n2_5 = torch.FloatTensor(gridPtsCam[4*part:,:]).cuda()
        # #print(n1.size())
        # #knnDist = torch.FloatTensor(n2.size()[0],1).cuda()
        # #for i in range(n2.size()[0]):
        # #    dist = torch.sum((n1 - n2[i,:])**2,1)
        # #    knnDist[i] = torch.min(dist)
        # #    knnDist[i] = torch.sqrt(knnDist[i])

        # #sum_1 = torch.t(torch.sum(n1**2, 1).repeat(n2.size()[0],1))
        # #sum_2 = torch.sum(n2**2, 1).repeat(n1.size()[0],1)
 
        # knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_1.size()[0],1))+torch.sum(n2_1**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_1)),0)
        # knnDist = torch.clamp(knnDist,min=0.0)
        # knnDist1 = torch.sqrt(knnDist)

        # knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_2.size()[0],1))+torch.sum(n2_2**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_2)),0)
        # knnDist = torch.clamp(knnDist,min=0.0)
        # knnDist2 = torch.sqrt(knnDist)

        # knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_3.size()[0],1))+torch.sum(n2_3**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_3)),0)
        # knnDist = torch.clamp(knnDist,min=0.0)
        # knnDist3 = torch.sqrt(knnDist)
        
        # knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_4.size()[0],1))+torch.sum(n2_4**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_4)),0)
        # knnDist = torch.clamp(knnDist,min=0.0)
        # knnDist4 = torch.sqrt(knnDist)
        
        # knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_5.size()[0],1))+torch.sum(n2_5**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_5)),0)
        # knnDist = torch.clamp(knnDist,min=0.0)
        # knnDist5 = torch.sqrt(knnDist)

        # knnDist = torch.cat((knnDist1,knnDist2,knnDist3,knnDist4,knnDist5))


        # Use 1-NN search to get TDF values
        # knnDist,knnIdx = spatial.KDTree(camPts).query(gridPtsCam)

        tree = cKDTree(camPts, leafsize=camPts.shape[0]+1)
        knnDist,knnIdx = tree.query(gridPtsCam, k=1, n_jobs=-1)


        TDFValues = knnDist/voxelMargin # truncate
        TDFValues[TDFValues > 1] = 1
        TDFValues = 1-TDFValues  # flip

        H = np.shape(gridPtsCamX)[0]
        W = np.shape(gridPtsCamX)[1]
        D = np.shape(gridPtsCamX)[2]

        # voxelGridTDF = np.reshape(TDFValues.cpu().data.numpy().T,(H,W,D)).T.reshape((1,H,W,D))
        voxelGridTDF = np.reshape(TDFValues.T,(H,W,D)).T.reshape((1,H,W,D))

        #print(voxelGridTDF[0,29,26])
        #voxelGridTDF = np.reshape(TDFValues,(None,H,W,D))

        ##############################################################


        if plot:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(camPts[np.where(camPts[:,0] != 0),0], camPts[np.where(camPts[:,0] != 0),1], camPts[np.where(camPts[:,0] != 0),2], s=5, c='b')
            
            ax.legend()
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            


            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # ax.scatter(gridPtsCam[:,0], gridPtsCam[:,1], gridPtsCam[:,2], s=5, c='b')
            
            # ax.legend()
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')


            #print(np.shape(gridPtsCamX))
            #print(np.shape(voxelGridTDF))
            a = np.linspace(0, knnDist.shape[0], knnDist.shape[0])
            b = np.linspace(0, len(np.reshape(camX, (-1))), len(np.reshape(camX, (-1))))
            c = np.linspace(0, len(np.reshape(camY, (-1))), len(np.reshape(camY, (-1))))
            d = np.linspace(0, len(np.reshape(camZ, (-1))), len(np.reshape(camZ, (-1))))

            plt.figure()
            plt.subplot(221)
            plt.title('knn distances')
            plt.scatter(a, knnDist/voxelMargin, s=5)
            plt.grid()

            plt.subplot(222)
            plt.title('tdf values')
            plt.scatter(a, TDFValues, s=5)
            plt.grid()

            plt.subplot(234)
            plt.title('camX')
            plt.scatter(b, np.reshape(camX, (-1)), s=5)
            plt.grid()

            plt.subplot(235)
            plt.title('camY')
            plt.scatter(c, np.reshape(camY, (-1)), s=5)
            plt.grid()

            plt.subplot(236)
            plt.title('camZ')
            plt.scatter(d, np.reshape(camZ, (-1)), s=5)
            plt.grid()




            plt.figure()
            plt.plot(np.reshape(voxelGridTDF, (-1)))

        if output:
            print()
            print()
            print()
            print()
            print('*********************************************')
            print('*********************************************')
            print()
            print('depthIm.shape............', depthIm.shape)
            print()
            print('---')
            print()
            print('lowBoundX................', lowBoundX)
            print()
            print('---')
            print()
            print('highBoundX...............', highBoundX)
            print()
            print('---')
            print()
            print('lowBoundY................', lowBoundY)
            print()
            print('---')
            print()
            print('highBoundY...............', highBoundY)
            print()
            print('---')
            print()
            print('depthPatch.shape.........', depthPatch.shape)
            print()
            print('---')
            print()
            print('mgrid....................', np.mgrid[lowBoundY:highBoundY, lowBoundX:highBoundX].shape)
            print()
            print('---')
            print()
            print('pixX:', pixX.shape)
            print(pixX)
            print()
            print('---')
            print()
            print('pixY:', pixY.shape)
            print(pixY)           
            print()
            print('---')
            print()
            print('camX:', camX.shape)
            print(camX)
            print()
            print('---')
            print()
            print('camY:', camY.shape)
            print(camY)
            print()
            print('---')
            print()
            print('camZ:', camZ.shape)
            print(camZ)
            print()
            print('---')
            print()
            print('ValidX:', ValidX.shape)
            print(ValidX)
            print()
            print('---')
            print()
            print('ValidY:', ValidY.shape)
            print(ValidY)
            print()
            print('---')
            print()
            print('ValidDepth:', ValidDepth.shape)
            print(ValidDepth)
            print()
            print('---')
            print()
            print('camPts:', camPts.shape)
            print(camPts)
            print()
            print('---')
            print()
            print('sigma TDF:', np.array([camX, camY, camZ]).shape)
            print(np.round(np.array([camX, camY, camZ]), 2))
            print()
            print('---')
            print()
            print('lowBoundX................', lowBoundX)
            print()
            print('---')
            print()
            print('highBoundX...............', highBoundX)
            print()
            print('---')
            print()
            print('lowBoundY................', lowBoundY)
            print()
            print('---')
            print()
            print('highBoundY...............', highBoundY)
            print()
            print('---')
            print()
            print('lowBoundZ................', lowBoundZ)
            print()
            print('---')
            print()
            print('highBoundZ...............', highBoundZ)
            print()
            print('---')
            print()
            print('gridPtsCamX:', gridPtsCamX.shape)
            print(gridPtsCamX)
            print()
            print('---')
            print()
            print('gridPtsCamY:', gridPtsCamY.shape)
            print(gridPtsCamY)
            print()
            print('---')
            print()
            print('gridPtsCamZ:', gridPtsCamZ.shape)
            print(gridPtsCamZ)
            print()
            print('---')
            print()
            print('gridPtsCam:', gridPtsCam.shape)
            print(gridPtsCam)
            print()
            print('---')
            print()
            print('knnDist:', knnDist.shape)
            print(knnDist)
            print()
            print('---')
            print()
            print('TDFValues:', TDFValues.shape)
            print(TDFValues)
            print()
            print('---')
            print()
            print('voxelGridTDF:', voxelGridTDF.shape)
            print(voxelGridTDF)
            print()
            print('*********************************************')
            print('*********************************************')
            print()
            print()
            print()
            print()
            
            print('np.shape(voxelGridTDF)')
            print(np.shape(voxelGridTDF))


        # print(TDFValues.cpu().data.numpy())

        return voxelGridTDF

def getPatchData_GPU(pointData,voxelGridPatchRadius,voxelSize,voxelMargin):

    output = False
    plot = False

    fx, fy, cx, cy = get_cam_intrinsics()

    depthIm = pointData['depthIm']
    depthIm[depthIm > 8] = 0


    # lowBoundX = int(np.min(pointData['bboxRangePixels'][1,:]))
    # highBoundX = int(np.max(pointData['bboxRangePixels'][1,:]))+1
    # lowBoundY = int(np.min(pointData['bboxRangePixels'][0,:]))
    # highBoundY = int(np.max(pointData['bboxRangePixels'][0,:]))+1


    lowBoundX = int(np.min(pointData['bboxRangePixels'][1,:]))
    highBoundX = int(np.max(pointData['bboxRangePixels'][1,:]))+1
    lowBoundY = int(np.min(pointData['bboxRangePixels'][0,:]))
    highBoundY = int(np.max(pointData['bboxRangePixels'][0,:]))+1

    depthPatch = depthIm[lowBoundX-1:highBoundX-1, lowBoundY-1:highBoundY-1]


    # Get TDF voxel grid local patches
    [pixX,pixY] = np.mgrid[lowBoundY:highBoundY, lowBoundX:highBoundX]
    pixX, pixY = pixX.T, pixY.T


    if pixX.shape != depthPatch.shape or pixY.shape != depthPatch.shape:
        if output:
            print()
            print()
            print('###################################')
            print('###### fatal dimension error ######')
            print('###################################')
            print()
            print()

        error_solution = np.asarray(np.zeros((1, window_size, window_size, window_size)))

        return error_solution

    else:


        camX = np.array((pixX-cx)*depthPatch/fx)
        camY = np.array(-(pixY-cy)*depthPatch/fy)
        camZ = np.array(depthPatch)

        

        # ValidX,ValidY = np.nonzero(depthPatch)[0], np.nonzero(depthPatch)[1]
        # ValidDepth = ValidX*np.shape(depthPatch)[1] + ValidY
        

        ValidX,ValidY = np.array(np.nonzero(depthPatch))
        ValidDepth = ValidX*np.shape(depthPatch)[1] + ValidY
        

        camPts = np.append(np.reshape(camX.T,(np.size(camX),1)),np.reshape(camY.T,(np.size(camY),1)),1)
        camPts = np.append(camPts,np.reshape(camZ.T,(np.size(camZ),1)),1)
        #camPts = np.array([camX,camY,camZ])
        #camPts = np.reshape(camPts,(np.size(depthPatch),3))
        # camPts = camPts[ValidDepth,:]



        #gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[(pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize, 
        #   np.float16(pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2):np.float16(pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize,
        #   (pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize]

        lowBoundX = pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2
        highBoundX = pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
        lowBoundY = pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2
        highBoundY = pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
        lowBoundZ = pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2
        highBoundZ = pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1



        gridPtsCamX, gridPtsCamY, gridPtsCamZ = np.mgrid[lowBoundX:highBoundX:voxelSize, lowBoundY:highBoundY:voxelSize, lowBoundZ:highBoundZ:voxelSize]

        gridPtsCam = np.append(np.reshape(gridPtsCamX.T,(np.size(gridPtsCamX),1)),np.reshape(gridPtsCamY.T,(np.size(gridPtsCamX),1)),1)
        gridPtsCam = np.append(gridPtsCam,np.reshape(gridPtsCamZ.T,(np.size(gridPtsCamZ),1)),1)


        n1 = torch.FloatTensor(camPts).cuda()


        part = int(np.round(gridPtsCam.shape[0]/3))

        n2_1 = torch.FloatTensor(gridPtsCam[:part,:]).cuda()
        n2_2 = torch.FloatTensor(gridPtsCam[part:2*part,:]).cuda()
        n2_3 = torch.FloatTensor(gridPtsCam[2*part:,:]).cuda()
        #print(n1.size())
        #knnDist = torch.FloatTensor(n2.size()[0],1).cuda()
        #for i in range(n2.size()[0]):
        #    dist = torch.sum((n1 - n2[i,:])**2,1)
        #    knnDist[i] = torch.min(dist)
        #    knnDist[i] = torch.sqrt(knnDist[i])

        #sum_1 = torch.t(torch.sum(n1**2, 1).repeat(n2.size()[0],1))
        #sum_2 = torch.sum(n2**2, 1).repeat(n1.size()[0],1)
        knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_1.size()[0],1))+torch.sum(n2_1**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_1)),0)
        knnDist1 = torch.sqrt(knnDist)

        knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_2.size()[0],1))+torch.sum(n2_2**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_2)),0)
        knnDist2 = torch.sqrt(knnDist)

        knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_3.size()[0],1))+torch.sum(n2_3**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_3)),0)
        knnDist3 = torch.sqrt(knnDist)

        knnDist = torch.cat((knnDist1,knnDist2,knnDist3))


        # Use 1-NN search to get TDF values
        # knnDist,knnIdx = spatial.KDTree(camPts).query(gridPtsCam)

        # tree = cKDTree(camPts, leafsize=camPts.shape[0]+1)
        # knnDist,knnIdx = tree.query(gridPtsCam, k=1, n_jobs=-1)


        TDFValues = knnDist/voxelMargin # truncate
        TDFValues[TDFValues > 1] = 1
        TDFValues = 1-TDFValues  # flip

        H = np.shape(gridPtsCamX)[0]
        W = np.shape(gridPtsCamX)[1]
        D = np.shape(gridPtsCamX)[2]

        voxelGridTDF = np.reshape(TDFValues.cpu().data.numpy().T,(H,W,D)).T.reshape((1,H,W,D))

        #print(voxelGridTDF[0,29,26])
        #voxelGridTDF = np.reshape(TDFValues,(None,H,W,D))

        ##############################################################


        if plot:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(camPts[np.where(camPts[:,0] != 0),0], camPts[np.where(camPts[:,0] != 0),1], camPts[np.where(camPts[:,0] != 0),2], s=5, c='b')
            
            ax.legend()
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            


            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # ax.scatter(gridPtsCam[:,0], gridPtsCam[:,1], gridPtsCam[:,2], s=5, c='b')
            
            # ax.legend()
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')


            #print(np.shape(gridPtsCamX))
            #print(np.shape(voxelGridTDF))
            a = np.linspace(0, knnDist.shape[0], knnDist.shape[0])
            b = np.linspace(0, len(np.reshape(camX, (-1))), len(np.reshape(camX, (-1))))
            c = np.linspace(0, len(np.reshape(camY, (-1))), len(np.reshape(camY, (-1))))
            d = np.linspace(0, len(np.reshape(camZ, (-1))), len(np.reshape(camZ, (-1))))

            plt.figure()
            plt.subplot(221)
            plt.title('knn distances')
            plt.scatter(a, knnDist/voxelMargin, s=5)
            plt.grid()

            plt.subplot(222)
            plt.title('tdf values')
            plt.scatter(a, TDFValues, s=5)
            plt.grid()

            plt.subplot(234)
            plt.title('camX')
            plt.scatter(b, np.reshape(camX, (-1)), s=5)
            plt.grid()

            plt.subplot(235)
            plt.title('camY')
            plt.scatter(c, np.reshape(camY, (-1)), s=5)
            plt.grid()

            plt.subplot(236)
            plt.title('camZ')
            plt.scatter(d, np.reshape(camZ, (-1)), s=5)
            plt.grid()




            plt.figure()
            plt.plot(np.reshape(voxelGridTDF, (-1)))

        if output:
            print()
            print()
            print()
            print()
            print('*********************************************')
            print('*********************************************')
            print()
            print('depthIm.shape............', depthIm.shape)
            print()
            print('---')
            print()
            print('lowBoundX................', lowBoundX)
            print()
            print('---')
            print()
            print('highBoundX...............', highBoundX)
            print()
            print('---')
            print()
            print('lowBoundY................', lowBoundY)
            print()
            print('---')
            print()
            print('highBoundY...............', highBoundY)
            print()
            print('---')
            print()
            print('depthPatch.shape.........', depthPatch.shape)
            print()
            print('---')
            print()
            print('mgrid....................', np.mgrid[lowBoundY:highBoundY, lowBoundX:highBoundX].shape)
            print()
            print('---')
            print()
            print('pixX:', pixX.shape)
            print(pixX)
            print()
            print('---')
            print()
            print('pixY:', pixY.shape)
            print(pixY)           
            print()
            print('---')
            print()
            print('camX:', camX.shape)
            print(camX)
            print()
            print('---')
            print()
            print('camY:', camY.shape)
            print(camY)
            print()
            print('---')
            print()
            print('camZ:', camZ.shape)
            print(camZ)
            print()
            print('---')
            print()
            print('ValidX:', ValidX.shape)
            print(ValidX)
            print()
            print('---')
            print()
            print('ValidY:', ValidY.shape)
            print(ValidY)
            print()
            print('---')
            print()
            print('ValidDepth:', ValidDepth.shape)
            print(ValidDepth)
            print()
            print('---')
            print()
            print('camPts:', camPts.shape)
            print(camPts)
            print()
            print('---')
            print()
            print('sigma TDF:', np.array([camX, camY, camZ]).shape)
            print(np.round(np.array([camX, camY, camZ]), 2))
            print()
            print('---')
            print()
            print('lowBoundX................', lowBoundX)
            print()
            print('---')
            print()
            print('highBoundX...............', highBoundX)
            print()
            print('---')
            print()
            print('lowBoundY................', lowBoundY)
            print()
            print('---')
            print()
            print('highBoundY...............', highBoundY)
            print()
            print('---')
            print()
            print('lowBoundZ................', lowBoundZ)
            print()
            print('---')
            print()
            print('highBoundZ...............', highBoundZ)
            print()
            print('---')
            print()
            print('gridPtsCamX:', gridPtsCamX.shape)
            print(gridPtsCamX)
            print()
            print('---')
            print()
            print('gridPtsCamY:', gridPtsCamY.shape)
            print(gridPtsCamY)
            print()
            print('---')
            print()
            print('gridPtsCamZ:', gridPtsCamZ.shape)
            print(gridPtsCamZ)
            print()
            print('---')
            print()
            print('gridPtsCam:', gridPtsCam.shape)
            print(gridPtsCam)
            print()
            print('---')
            print()
            print('knnDist:', knnDist.shape)
            print(knnDist)
            print()
            print('---')
            print()
            print('TDFValues:', TDFValues.shape)
            print(TDFValues)
            print()
            print('---')
            print()
            print('voxelGridTDF:', voxelGridTDF.shape)
            print(voxelGridTDF)
            print()
            print('*********************************************')
            print('*********************************************')
            print()
            print()
            print()
            print()
            
            print('np.shape(voxelGridTDF)')
            print(np.shape(voxelGridTDF))


        # print(TDFValues.cpu().data.numpy())

        return voxelGridTDF

def doublette(whichTrees):
    
    ############################
    depth_limit = 8.
    cam_distance_threshold = 3.5
    point_dist_threshold = 1.5
    gamma_tolerance = .02
    frame_size = voxelGridPatchRadius
    max_iter = 1

    plots = False
    output = False

    match_found = 0
    non_match_found = 0
    first_img_found = 0
    ############################

    fx, fy, cx, cy = get_cam_intrinsics()
    drone_cam_translation = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., .46], [0., 0., 0., 1.]])
    # print('fx, fy, cx, cy...............', fx, fy, cx, cy)
    

    if output:
        print()
        print('finding initial camera position with visible tree..........................')
        print()


    while match_found < 1:

        gamma_loop = 0
        frustum_loop = 0
        bbox_loop = 0

        first_img_found = 0

        while first_img_found < 1:
        
            '''pick a random image number for snapshot1'''
            if whichTrees == 'random':

                random_tree_kind = np.random.randint(len(trees))
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = np.random.randint(num_tree_folders)
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            if whichTrees == 'debug':

                random_tree_kind = 0
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = random_tree_debug
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            if whichTrees == 'debug_tejaswi':

                random_tree_kind = 0
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = 1
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            if whichTrees == 'Cherry':

                random_tree_kind = 0
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = np.random.randint(1,num_tree_folders)
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            if whichTrees == 'reconstruction_trial':

                random_tree_kind = 0
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = 1
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            if whichTrees == 'KoreanStewartia':

                random_tree_kind = 1
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = np.random.randint(1,num_tree_folders)
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            '''find total number of frames in respective folder'''
            data_table = np.asarray(pd.read_table(random_tree_path + 'poses.txt' , sep='\s', header=0, index_col=False, engine='python'))
            num_frames = np.max(data_table[:,0])

            abc = np.squeeze(np.array(np.where(data_table[:,7] == 1)))
            mean_x = np.mean(data_table[abc[0]:abc[-1],1])
            mean_y = np.mean(data_table[abc[0]:abc[-1],2])

            drone_trajectory_z = data_table[:,3] * -1
            drone_trajectory_x = (data_table[:,1] - mean_x)
            drone_trajectory_y = (data_table[:,2] - mean_y)
            drone_trajectory = np.array([drone_trajectory_x, drone_trajectory_y, drone_trajectory_z])
            drone_trajectory_x = drone_trajectory[0, :]
            drone_trajectory_y = drone_trajectory[1, :]
            drone_trajectory_z = drone_trajectory[2, :]


            '''find random initial frame'''
            snapshot1_index = np.random.randint(num_frames)
            if whichTrees == 'debug':
                snapshot1_index = snapshot1_index_debug

            if whichTrees == 'reconstruction_trial':
                snapshot1_index = 5


            '''load respective depth img'''
            file_index1 = file_index(snapshot1_index)

            depthImg1_raw = convert_pfm(random_tree_path + file_index1 + 'pl.pfm')
            '''delete too far away points'''
            depthImg1 = depthImg1_raw
            depthImg1[depthImg1_raw > depth_limit] = 0 


            '''check if there is a tree'''
            if np.array(np.nonzero(depthImg1[frame_size:-frame_size,frame_size:-frame_size])).shape[1] > 1000:

                first_img_found = 1


                '''choose random point on tree'''
                rand_row, rand_col = 0, 0

                while np.absolute(cy - rand_row) > (cy - frame_size) or np.absolute(cx - rand_col) > (cx - frame_size):
                    random_spot1 = np.random.randint(0, np.array(np.nonzero(depthImg1)).shape[1])
                    rand_row, rand_col = np.array(np.nonzero(depthImg1))[:,random_spot1]

                if whichTrees == 'debug':

                    rand_row, rand_col = rand_row_debug, rand_col_debug      

                '''get 1st cam position and quaternions'''
                cam_position1 = get_cam_position(snapshot1_index, random_tree_path)
                qx1 = cam_position1[3]
                qy1 = cam_position1[4]
                qz1 = cam_position1[5]
                if qz1 == 0.:
                    qz1 = 1e-04
                qw1 = cam_position1[6]

                quaternion_cam1 = Quaternion(qw1, qx1, qy1, qz1)


                x1, y1, z1 = cam_position1[:3]
                x1 = x1 - mean_x
                y1 = y1 - mean_y
                z1 = -1 * z1
                drone1position = np.transpose(np.array([x1, y1, z1]))
                lens1position = drone1position + quaternion_cam1.rotate(np.array([drone_cam_translation[2,3], 0, 0]))


        match_loop = 0

        if gamma_loop > max_iter or frustum_loop > max_iter or bbox_loop > max_iter:
            break
        
        while match_found < 1:


            if gamma_loop > max_iter or frustum_loop > max_iter or bbox_loop > max_iter:
                break


            '''find 2nd camera position in proximity'''
            cam_distance = 100.

            match_loop += 1

            while cam_distance > cam_distance_threshold:

                '''choose random and different 2nd point'''
                snapshot2_index = snapshot1_index
                while snapshot2_index == snapshot1_index:
                    snapshot2_index = np.random.randint(num_frames)
                if whichTrees == 'debug':
                    snapshot2_index = snapshot2_index_debug
                if whichTrees == 'reconstruction_trial':
                    snapshot2_index = 6

                file_index2 = file_index(snapshot2_index)


                '''get 2nd cam position and quaternion'''
                cam_position2 = get_cam_position(snapshot2_index, random_tree_path)
                qx2 = cam_position2[3]
                qy2 = cam_position2[4]
                qz2 = cam_position2[5]
                if qz2 == 0.:
                    qz2 = 1e-04
                qw2 = cam_position2[6]

                quaternion_cam2 = Quaternion(qw2, qx2, qy2, qz2)

          
                x2, y2, z2 = cam_position2[:3]
                x2 = x2 - mean_x
                y2 = y2 - mean_y
                z2 = -1 * z2
                drone2position = np.transpose(np.array([x2, y2, z2]))

                lens2position = drone2position + quaternion_cam2.rotate(np.array([drone_cam_translation[2,3], 0, 0]))


                '''load respective depth img'''
                depthImg2_raw = convert_pfm(random_tree_path + file_index2 + 'pl.pfm')
                depthImg2 = depthImg2_raw
                depthImg2[depthImg2_raw > depth_limit] = 0

                # '''load the segmentation'''
                # segmentation2 = imageio.imread(random_tree_path + file_index2 + 'seg.ppm')
                # segmentation2 = np.round(np.mean(segmentation2, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation2,2), (-1)))), 1)

                '''check if relative distance under threshold'''
                cam_distance = dist_calc(lens1position, lens2position)


            if gamma_loop > max_iter or frustum_loop > max_iter or bbox_loop > max_iter:
                break


            '''Compute transformtation'''
            translation = np.transpose(np.array([x1, y1, z1]) - np.array([x2, y2, z2]))

            '''compute sigma1 -> point coordinates'''
            cam1_2_point_distance = depthImg1[rand_row, rand_col]
            d1 = cam1_2_point_distance

            '''translate row, col to cam-coords'''
            u1, v1 = rand_col, 2 * cy - rand_row

            alpha1 = (u1 - cx) * d1 / fx
            beta1 = (v1 - cy) * d1 / fy
            gamma1 = d1

            sigma1 = np.transpose(np.array([alpha1, beta1, gamma1, 1.]))

            PointRealWorld = drone1position + quaternion_cam1.rotate(np.array([gamma1 + .46, -alpha1, beta1]))

            # lens2point1 = PointRealWorld - lens1position
            # lens2point2 = PointRealWorld - lens2position


            lens2point1 = PointRealWorld - drone1position
            lens2point2 = PointRealWorld - drone2position


            lens2point1_vector = np.array([
                [lens1position[0], PointRealWorld[0]],
                [lens1position[1], PointRealWorld[1]],
                [lens1position[2], PointRealWorld[2]]])
            lens2point2_vector = np.array([
                [lens2position[0], PointRealWorld[0]],
                [lens2position[1], PointRealWorld[1]],
                [lens2position[2], PointRealWorld[2]]])

            newSigma = quaternion_cam2.inverse.rotate(lens2point2)

            alpha2 = -1 * newSigma[1]
            beta2 = newSigma[2]
            gamma2 = newSigma[0] - .46

            sigma2 = np.transpose(np.array([alpha2, beta2, gamma2, 1.]))

            PointRealWorld2 = lens2position + quaternion_cam2.rotate(np.array([gamma2, -alpha2, beta2]))


            d2 = gamma2

            u2 = (alpha2 * fx / d2) + cx
            v2 = (beta2 * fy / d2) + cy

            if np.abs(alpha2 * fx / d2) + frame_size > cx or np.abs(beta2 * fy / d2) + frame_size > cy:
                if output:
                    print('error: not in camera frustum')
                frustum_loop += 1
                continue

            # if np.abs(alpha2 * fx / d2) + frame_size < cx and np.abs(beta2 * fy / d2) + frame_size < cy:
            #     match_found = 1

            col2 = int(np.round(u2))
            row2 = int(np.round(2 * cy - v2))

            # point_label2 = segmentation2[row2, col2]


            if np.abs(depthImg2[row2, col2] - gamma2) > gamma_tolerance:
                gamma_loop += 1
                if output:
                    print('error: depth does not match gamma')
                continue



            p1_bboxCornersCam = bbox_corners(sigma1[:3])
            p2_bboxCornersCam = bbox_corners(sigma2[:3])

            p1_bboxCorners_test, p2_bboxCorners_test = [], []
            for j in range(8):
                p1_bboxCorners_test.append(lens1position + quaternion_cam1.rotate(np.array([p1_bboxCornersCam[2,j], -p1_bboxCornersCam[0,j], p1_bboxCornersCam[1,j]])))
                p2_bboxCorners_test.append(lens2position + quaternion_cam2.rotate(np.array([p2_bboxCornersCam[2,j], -p2_bboxCornersCam[0,j], p2_bboxCornersCam[1,j]])))

            p1_bboxCorners_test = np.asarray(p1_bboxCorners_test)
            p2_bboxCorners_test = np.asarray(p2_bboxCorners_test)

            
            u_BBox1 = np.round((p1_bboxCornersCam[0,:] * fx / p1_bboxCornersCam[2,:]) + cx)
            v_BBox1 = 2*cy-np.round((p1_bboxCornersCam[1,:] * fy / p1_bboxCornersCam[2,:]) + cy)

            bboxPixX1 = np.array([np.int(u1-max(np.abs(u1-u_BBox1))), np.int(u1+max(np.abs(u1-u_BBox1)))])
            # bboxPixY1 = np.array([v1-max(np.abs(v1-v_BBox1)), v1+max(np.abs(v1-v_BBox1))])
            bboxPixY1 = np.array([np.int(rand_row-max(np.abs(rand_row-v_BBox1))), np.int(rand_row+max(np.abs(rand_row-v_BBox1)))])


            u_BBox2 = np.round((p2_bboxCornersCam[0,:] * fx / p2_bboxCornersCam[2,:]) + cx)
            v_BBox2 = 2*cy-np.round((p2_bboxCornersCam[1,:] * fy / p2_bboxCornersCam[2,:]) + cy)

            bboxPixX2 = np.array([np.int(u2-max(np.abs(u2-u_BBox2))), np.int(u2+max(np.abs(u2-u_BBox2)))])
            # bboxPixY2 = np.array([v2-max(np.abs(v2-v_BBox2)), v2+max(np.abs(v2-v_BBox2))])
            bboxPixY2 = np.array([np.int(row2-max(np.abs(row2-v_BBox2))), np.int(row2+max(np.abs(row2-v_BBox2)))])


            if np.any(bboxPixX1 <= 0) or np.any(bboxPixX1 > 2*cx) or np.any(bboxPixY1 <= 0) or np.any(bboxPixY1 > 2*cy) :
                match_found = 0
                bbox_loop += 1
                if output:
                    print('error: not in bbox frustum')
                continue

            if np.any(bboxPixX2 <= 0) or np.any(bboxPixX2 > 2*cx) or np.any(bboxPixY2 <= 0) or np.any(bboxPixY2 > 2*cy) :
                if output:
                    print('error: not in bbox frustum')
                match_found = 0
                bbox_loop += 1
                continue

            p1_bboxRangePixels = np.array([[bboxPixX1],[bboxPixY1]])
            p2_bboxRangePixels = np.array([[bboxPixX2],[bboxPixY2]])


            match_found = 1


            if gamma_loop > max_iter or frustum_loop > max_iter or bbox_loop > max_iter:
                break


        if gamma_loop > max_iter or frustum_loop > max_iter or bbox_loop > max_iter:
            continue



    '''load the segmentation'''
    segmentation1 = imageio.imread(random_tree_path + file_index1 + 'seg.ppm')
    # segmentation1 = np.round(np.mean(segmentation1, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation1,2), (-1)))), 3)
    segmentation1 = np.round(np.sum(segmentation1, 2), 3)

    segmentation2 = imageio.imread(random_tree_path + file_index2 + 'seg.ppm')
    # segmentation2 = np.round(np.mean(segmentation2, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation2,2), (-1)))), 3)
    segmentation2 = np.round(np.sum(segmentation2, 2), 3)


    '''label point'''
    point_label1 = segmentation1[rand_row, rand_col]

    point_label2 = segmentation2[row2, col2]



    p1_info = {'framePath': random_tree_path + file_index1,
               'pixelCoords': np.array([rand_col, rand_row]),
               'camCoords': sigma1[:3],
               'label': point_label1,
               'cam_position': cam_position1,
               'PointRealWorld': PointRealWorld}

    p2_info = {'framePath': random_tree_path + file_index2,
               'pixelCoords': np.array([col2, row2]),
               'camCoords': sigma2[:3],
               'label': point_label2,
               'cam_position': cam_position2,
               'PointRealWorld': PointRealWorld}




    p1 = {'bboxCornersCam': p1_bboxCornersCam,
          'bboxRangePixels': p1_bboxRangePixels, 
          'framePath': random_tree_path + file_index1,
          'pixelCoords': np.array([rand_col, rand_row]),
          'camCoords': sigma1[:3],
          'label': point_label1,
          'depthIm': depthImg1_raw}

    p2 = {'bboxCornersCam': p2_bboxCornersCam,
          'bboxRangePixels': p2_bboxRangePixels, 
          'framePath': random_tree_path + file_index2,
          'pixelCoords': np.array([col2, row2]),
          'camCoords': sigma2[:3],
          'label': point_label2,
          'depthIm': depthImg2_raw}


    while non_match_found < 1:

        point2point_dist = 0.

        while point2point_dist < point_dist_threshold:

            '''find random initial frame'''
            snapshot3_index = np.random.randint(num_frames)

            if whichTrees == 'reconstruction_trial':
                    snapshot3_index = np.random.choice(np.array([5,6]))

            
            '''load respective depth img'''
            file_index3 = file_index(snapshot3_index)

            depthImg3_raw = convert_pfm(random_tree_path + file_index3 + 'pl.pfm')
            depthImg3 = depthImg3_raw
            '''delete too far away points'''
            depthImg3[depthImg3_raw > depth_limit] = 0 


            '''check if there is a tree'''
            if np.array(np.nonzero(depthImg3[frame_size:-frame_size,frame_size:-frame_size])).shape[1] > 10:


                random_spot3 = np.random.randint(0, np.array(np.nonzero(depthImg3[frame_size:-frame_size,frame_size:-frame_size])).shape[1])
                row3, col3 = np.array(np.nonzero(depthImg3))[:,random_spot3]

                '''get 1st cam position and quaternions'''
                cam_position3 = get_cam_position(snapshot3_index, random_tree_path)
                qx3 = cam_position3[3]
                qy3 = cam_position3[4]
                qz3 = cam_position3[5]
                if qz3 == 0.:
                    qz3 = 1e-04
                qw3 = cam_position3[6]

                quaternion_cam3 = Quaternion(qw3, qx3, qy3, qz3)

                x3, y3, z3 = cam_position3[:3]
                x3 = x3 - mean_x
                y3 = y3 - mean_y
                z3 = -1 * z3
                drone3position = np.transpose(np.array([x3, y3, z3]))
                lens3position = drone3position + quaternion_cam3.rotate(np.array([drone_cam_translation[2,3], 0, 0]))


                d3 = depthImg3[row3, col3]



                '''translate row, col to cam-coords'''
                u3, v3 = col3, 2 * cy - row3

                alpha3 = (u3 - cx) * d3 / fx
                beta3 = (v3 - cy) * d3 / fy
                gamma3 = d3

                sigma3 = np.transpose(np.array([alpha3, beta3, gamma3, 1.]))

                PointRealWorld3 = drone3position + quaternion_cam3.rotate(np.array([gamma3 + .46, -alpha3, beta3]))

                lens2point3 = PointRealWorld3 - drone3position

                lens2point3_vector = np.array([
                    [lens3position[0], PointRealWorld3[0]],
                    [lens3position[1], PointRealWorld3[1]],
                    [lens3position[2], PointRealWorld3[2]]])


                point2point_dist = dist_calc(PointRealWorld3, PointRealWorld)



                p3_bboxCornersCam = bbox_corners(sigma3[:3])

                p3_bboxCorners_test = []
                for j in range(8):
                    p3_bboxCorners_test.append(lens3position + quaternion_cam3.rotate(np.array([p3_bboxCornersCam[2,j], -p3_bboxCornersCam[0,j], p3_bboxCornersCam[1,j]])))

                p3_bboxCorners_test = np.asarray(p3_bboxCorners_test)


                u_BBox3 = np.round((p3_bboxCornersCam[0,:] * fx / p3_bboxCornersCam[2,:]) + cx)
                v_BBox3 = 2*cy-np.round((p3_bboxCornersCam[1,:] * fy / p3_bboxCornersCam[2,:]) + cy)

                bboxPixX3 = np.array([col3-max(np.abs(col3-u_BBox3)), col3+max(np.abs(col3-u_BBox3))])
                bboxPixY3 = np.array([row3-max(np.abs(row3-v_BBox3)), row3+max(np.abs(row3-v_BBox3))])
                
                p3_bboxRangePixels = np.array([[bboxPixX3],[bboxPixY3]])




                lowBoundX = int(np.min(p3_bboxRangePixels))
                highBoundX = int(np.max(p3_bboxRangePixels))+1
                lowBoundY = int(np.min(p3_bboxRangePixels))
                highBoundY = int(np.max(p3_bboxRangePixels))+1

                if lowBoundX < 1 or lowBoundY < 1 or highBoundX > 2*cx or highBoundY > 2*cy:
                    match_found = 0
                    if output:
                        print('error: bounds not in frame')
                    continue

                if np.any(bboxPixX3 <= 0) or np.any(bboxPixX3 > 2*cx) or np.any(bboxPixY3 <= 0) or np.any(bboxPixY3 > 2*cy) :
                    match_found = 0
                    if output:
                        print('error: point 3 not in bbox frustum')
                    continue

                else:
                    non_match_found = 1
                    if output:
                        print('3rd one fully loaded!')


    '''load the segmentation'''
    segmentation3 = imageio.imread(random_tree_path + file_index3 + 'seg.ppm')
    # segmentation3 = np.round(np.mean(segmentation3, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation3,2), (-1)))), 1)
    segmentation3 = np.round(np.sum(segmentation3, 2), 3)



    p3_bboxCorners_test = []
    for j in range(8):
        p3_bboxCorners_test.append(lens3position + quaternion_cam3.rotate(np.array([p3_bboxCornersCam[2,j], -p3_bboxCornersCam[0,j], p3_bboxCornersCam[1,j]])))
    p3_bboxCorners_test = np.asarray(p3_bboxCorners_test)



    '''label point'''
    point_label3 = segmentation3[row3, col3]


    p3_info = {'framePath': random_tree_path + file_index3,
               'pixelCoords': np.array([col3, row3]),
               'camCoords': sigma3[:3],
               'label': point_label3,
               'cam_position': cam_position3,
               'PointRealWorld': PointRealWorld3}



    p3 = {'bboxCornersCam': p3_bboxCornersCam,
          'bboxRangePixels': p3_bboxRangePixels, 
          'framePath': random_tree_path + file_index3,
          'pixelCoords': np.array([col3, row3]),
          'camCoords': sigma3[:3],
          'label': point_label3,
          'depthIm': depthImg3_raw}


    p1_voxelGridTDF = getPatchData(p1,voxelGridPatchRadius,voxelSize,voxelMargin)
    if output:
        print()
        print('1st TDF ==========> success!')
        print(p1_voxelGridTDF.ndim)
    p2_voxelGridTDF = getPatchData(p2,voxelGridPatchRadius,voxelSize,voxelMargin)
    if output:
        print()
        print('2nd TDF ==========> success!')    
    p3_voxelGridTDF = getPatchData(p3,voxelGridPatchRadius,voxelSize,voxelMargin)
    if output:
        print()
        print('3rd TDF ==========> success!')



    if output:
        # print('num_frames...............', num_frames)
        # print('random_tree_kind.........', random_tree_kind)
        # print('random_tree_kind_str.....', random_tree_kind_str)
        # print('random_tree..............', random_tree)
        # print('random_tree_path.........', random_tree_path)
        # print('index1...................', snapshot1_index)
        print(random_tree_path + file_index1 + 'pl.pfm')
        # print('rand_row, rand_col.......', rand_row, rand_col)
        # print('label 1st point..........', point_label1)
        print('lens1position............', np.round(lens1position, 2))
        print()
        print('=======================================================> initial frame found')
        print()
        print(random_tree_path + file_index2 + 'pl.pfm')
        print('lens2position............', np.round(lens2position, 2))
        print('cam_distance.............', np.round(cam_distance, 2))
        print()
        print('================================================> 2nd frame found')
        print()
        print(random_tree_path + file_index1 + 'pl.pfm')
        print()
        print('col1, row1...............', rand_col, rand_row)
        print('col2, row2...............', col2, row2)
        print('col3, row3...............', col3, row3)
        print()
        print('sigma1:', np.round(sigma1[:3],1))
        print('sigma2:', np.round(sigma2[:3],1))
        print('sigma3:', np.round(sigma3[:3],1))
        print()
        print('PointRealWorld1')
        print(np.round(PointRealWorld,1))
        print('PointRealWorld2')
        print(np.round(PointRealWorld2,1))
        print('PointRealWorld3')
        print(np.round(PointRealWorld3,1))
        # print('num_frames...............', num_frames)
        # print('random_tree_kind.........', random_tree_kind)
        # print('random_tree_kind_str.....', random_tree_kind_str)
        # print('random_tree..............', random_tree)
        # print('random_tree_path.........', random_tree_path)
        # print('index1...................', snapshot1_index)
        print()
        print('=======================================================> third frame found')
        print()

    if plots:

        # '''load respective depth img'''
        # depthImg2 = convert_pfm(random_tree_path + file_index2 + 'pl.pfm')
        # depthImg2[depthImg2 > depth_limit] = 0

        print('depth-------', depthImg2[row2, col2])


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, y1, z1, c='b')
        ax.scatter(lens1position[0], lens1position[1], lens1position[2], c='b', label='inital cam')

        ax.scatter(x2, y2, z2, c='g')
        ax.scatter(lens2position[0], lens2position[1], lens2position[2], c='g', label='match cam')

        ax.scatter(x3, y3, z3, c='r')
        ax.scatter(lens3position[0], lens3position[1], lens3position[2], c='r', label='non-match cam')

        ax.scatter(PointRealWorld[0], PointRealWorld[1], PointRealWorld[2], c='b', s=10, label='point1')
        ax.scatter(PointRealWorld2[0], PointRealWorld2[1], PointRealWorld2[2], c='g', s=10, label='point2')
        ax.scatter(PointRealWorld3[0], PointRealWorld3[1], PointRealWorld3[2], c='r', s=10, label='point3')

        ax.scatter(p1_bboxCorners_test[:,0], p1_bboxCorners_test[:,1], p1_bboxCorners_test[:,2], c='b', s=10, label='bbox1')
        ax.scatter(p2_bboxCorners_test[:,0], p2_bboxCorners_test[:,1], p2_bboxCorners_test[:,2], c='g', s=10, label='bbox2')
        ax.scatter(p3_bboxCorners_test[:,0], p3_bboxCorners_test[:,1], p3_bboxCorners_test[:,2], c='r', s=10, label='bbox3')

        ax.plot(lens2point1_vector[0], lens2point1_vector[1], lens2point1_vector[2], c='b', label='sight1')
        ax.plot(lens2point2_vector[0], lens2point2_vector[1], lens2point2_vector[2], c='g', label='sight2')
        ax.plot(lens2point3_vector[0], lens2point3_vector[1], lens2point3_vector[2], c='r', label='sight3')

        ax.plot(drone_trajectory_x, drone_trajectory_y, drone_trajectory_z, c='y', label='drone flight')

        ax.plot(np.zeros(2), np.zeros(2), np.linspace(0,8,2), c='k', label='tree')
        
        ax.legend()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)



        testimage1 = imageio.imread(random_tree_path + file_index1 + '.ppm')
        testimage2 = imageio.imread(random_tree_path + file_index2 + '.ppm')
        testimage3 = imageio.imread(random_tree_path + file_index3 + '.ppm')

        plt.figure(figsize=(10,10))
        plt.subplot(231)
        plt.imshow(depthImg1)
        # plt.scatter(rand_col, rand_row, c="r")
        plt.scatter(rand_col, rand_row, 80, 'w', 'x')
        #plt.scatter(np.array(np.nonzero(depthImg1))[1,:],np.array(np.nonzero(depthImg1))[0,:])
        plt.title('considered point:' + str(point_label1))
        plt.grid()

        plt.subplot(232)
        plt.imshow(depthImg2)
        plt.title('corresponding view point')
        plt.scatter(col2, row2, 80, 'w', 'x')
        # # plt.scatter(u2, v2, 80, 'y', 'x')
        # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
        # # plt.scatter(pixX, pixY, 80, 'r', 'x')
        plt.grid()

        plt.subplot(233)
        plt.imshow(depthImg3)
        plt.title('non-match')
        plt.scatter(col3, row3, 80, 'w', 'x')
        # # plt.scatter(u2, v2, 80, 'y', 'x')
        # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
        # # plt.scatter(pixX, pixY, 80, 'r', 'x')
        plt.grid()

        plt.subplot(234)
        plt.imshow(testimage1)
        plt.title('image 1')
        plt.scatter(rand_col, rand_row, 80, 'r', 'x')
        plt.grid()

        plt.subplot(235)
        plt.imshow(testimage2)
        plt.title('image 2')
        plt.scatter(col2, row2, 80, 'r', 'x')
        # # plt.scatter(u2, v2, 80, 'y', 'x')
        # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
        # # plt.scatter(pixX, pixY, 80, 'r', 'x')
        plt.grid()

        plt.subplot(236)
        plt.imshow(testimage3)
        plt.title('image 2')
        plt.scatter(col3, row3, 80, 'r', 'x')
        # # plt.scatter(u2, v2, 80, 'y', 'x')
        # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
        # # plt.scatter(pixX, pixY, 80, 'r', 'x')
        plt.grid()

        plt.show()


    return p1_voxelGridTDF, p2_voxelGridTDF, p3_voxelGridTDF, point_label1, p1_info, p2_info, p3_info

def TDF_train_loader_no_labels(batchsize):

    counter = 0

    tdf1 = np.zeros((batchsize, window_size, window_size, window_size, 1))
    tdf2 = np.zeros((batchsize, window_size, window_size, window_size, 1))
    tdf3 = np.zeros((batchsize, window_size, window_size, window_size, 1))

    error_solution = np.zeros((batch_size_train, window_size, window_size, window_size, 1))

    while counter < batchsize:

        p1, p2, p3, label1, p1_info, p2_info, p3_info = doublette('Cherry')

        if np.mean(np.reshape(p1, (-1))) != 0. and np.mean(np.reshape(p2, (-1))) != 0. and np.mean(np.reshape(p1, (-1))) != 0.:

            tdf1[counter, :,:,:,0] = p1
            tdf2[counter, :,:,:,0] = p2
            tdf3[counter, :,:,:,0] = p3

            counter += 1

    return tdf1, tdf2, tdf3

def TDF_train_loader(batchsize):

    counter_leaves = 0
    counter_non_leaves = 0

    tdf1L = np.zeros((int(batchsize/2), window_size, window_size, window_size, 1))
    tdf2L = np.zeros((int(batchsize/2), window_size, window_size, window_size, 1))
    tdf3L = np.zeros((int(batchsize/2), window_size, window_size, window_size, 1))

    tdf1NL = np.zeros((int(batchsize/2), window_size, window_size, window_size, 1))
    tdf2NL = np.zeros((int(batchsize/2), window_size, window_size, window_size, 1))
    tdf3NL = np.zeros((int(batchsize/2), window_size, window_size, window_size, 1))

    while counter_leaves < int(batchsize/2) or counter_non_leaves < int(batchsize/2):

        p1, p2, p3, label1 = doublette('Cherry')

        if np.mean(np.reshape(p1, (-1))) != 0. and np.mean(np.reshape(p2, (-1))) != 0. and np.mean(np.reshape(p1, (-1))) != 0.:

            if label1 > 400 and counter_leaves < int(batchsize/2):

                tdf1L[counter_leaves, :,:,:,0] = p1
                tdf2L[counter_leaves, :,:,:,0] = p2
                tdf3L[counter_leaves, :,:,:,0] = p3

                counter_leaves += 1

            if label1 < 400 and counter_non_leaves < int(batchsize/2):

                tdf1NL[counter_non_leaves, :,:,:,0] = p1
                tdf2NL[counter_non_leaves, :,:,:,0] = p2
                tdf3NL[counter_non_leaves, :,:,:,0] = p3

                counter_non_leaves += 1

    tdf1 = np.append(tdf1L, tdf1NL, axis=0)
    tdf2 = np.append(tdf2L, tdf2NL, axis=0)
    tdf3 = np.append(tdf3L, tdf3NL, axis=0)
    
    print('data assembly successful')
    
    return tdf1_train, tdf2_train, tdf3_train    

def TDF_recon_loader(batchsize):

    counter = 0

    tdf1 = np.zeros((int(batchsize), window_size, window_size, window_size, 1))
    tdf2 = np.zeros((int(batchsize), window_size, window_size, window_size, 1))
    tdf3 = np.zeros((int(batchsize), window_size, window_size, window_size, 1))

    p1data, p2data, p3data = [],[], []

    while counter < int(batchsize):

        p1, p2, p3, label1, p1_info, p2_info, p3_info = doublette('reconstruction_trial')

        if np.mean(np.reshape(p1, (-1))) != 0. and np.mean(np.reshape(p2, (-1))) != 0. and np.mean(np.reshape(p1, (-1))) != 0.:

            tdf1[counter, :,:,:,0] = p1
            tdf2[counter, :,:,:,0] = p2
            tdf3[counter, :,:,:,0] = p3

            p1data.append(p1_info)
            p2data.append(p2_info)
            p3data.append(p3_info)

            print('iteration:', counter)

            counter += 1

    return tdf1, p1data, tdf2, p2data, tdf3, p3data

def analyse(matches, non_matches, threshold):

    tp = matches < threshold
    fn = matches >= threshold
    tn = non_matches >= threshold
    fp = non_matches < threshold

    tp = np.sum(tp.astype(int))
    fn = np.sum(fn.astype(int))
    tn = np.sum(tn.astype(int))
    fp = np.sum(fp.astype(int))

    recall = np.round(100 * tp / (tp + fn), 2)
    err_rate =  np.round(100 * fp / (fp + tn), 2)

    return recall, err_rate

def feeder(logTDF, logFile, epoch, batchsize, tdf1, tdf2, tdf3):
    


    loggerlen = 1

    while loggerlen > 0:

        
        if len(logTDF) <= batchsize:
      
            if len(logFile) == 0:
                
                logFile = list(np.arange(numfiles+1))
                logFile.pop(0)

                epoch += 1
                        
            FileIndex = np.random.choice(np.asarray(logFile))

            print('load tdfs from file', FileIndex)
            tdf1 = np.load(path + 'tdf1' + str(FileIndex) + '.npy')
            tdf2 = np.load(path + 'tdf2' + str(FileIndex) + '.npy')
            tdf3 = np.load(path + 'tdf3' + str(FileIndex) + '.npy')


            logFile.pop(int(np.asarray(np.where(np.asarray(logFile) == FileIndex))))
          
            logTDF = list(np.arange(tdf_size))


        rand_samples = np.random.choice(np.asarray(logTDF), batchsize, replace=False)


        p1 = tdf1[rand_samples,:]
        p2 = tdf2[rand_samples,:]
        p3 = tdf3[rand_samples,:]

    
        logger = np.asarray(np.where(np.isnan(p1).astype(np.int32) == 1))[0,:]
        logger = np.append(logger, np.asarray(np.where(np.isnan(p2).astype(np.int32) == 1))[0,:])
        logger = np.append(logger, np.asarray(np.where(np.isnan(p3).astype(np.int32) == 1))[0,:])

        loggerlen = len(logger)


        for j in range(len(rand_samples)):

            logTDF.pop(int(np.asarray(np.where(np.asarray(logTDF) == rand_samples[j]))))


    return p1, p2, p3, logTDF, logFile, epoch, tdf1, tdf2, tdf3

def feeder_val(batchsize, logVal):

    valIdx = np.random.choice(np.asarray(logVal), batchsize, replace=False)

    for j in range(len(valIdx)):

        logVal.pop(int(np.asarray(np.where(np.asarray(logVal) == valIdx[j]))))

    p1 = tdf1val[valIdx,:]
    p2 = tdf2val[valIdx,:]
    p3 = tdf3val[valIdx,:]

    return p1, p2, p3, logVal

def feeder_recon(batchsize, logRec, combos):

    recIdx = np.random.choice(np.asarray(logRec), batchsize, replace=False)

    for j in range(len(recIdx)):

        logRec.pop(int(np.asarray(np.where(np.asarray(logRec) == recIdx[j]))))

    p1 = tdf1_rec[combos[0,recIdx],:]
    info1 = p1_info[combos[0,recIdx]]

    p2 = tdf2_rec[combos[1,recIdx],:]
    info2 = p2_info[combos[1,recIdx]]

    p3 = tdf3_rec[combos[2,recIdx],:]
    info3 = p3_info[combos[2,recIdx]]


    return p1, info1, p2, info2, p3, info3, logRec

def feeder_old(logTDF, logFile, epoch, batchsize):

    loggerlen = 1

    while loggerlen > 0:

        
        if len(logTDF) <= batchsize:
            
            logTDF = list(np.arange(tdf1.shape[0]))
            
            epoch += 1
        

        rand_samples = np.random.choice(np.asarray(logTDF), batchsize, replace=False)
        
        
        p1 = tdf1[rand_samples,:]
        p2 = tdf2[rand_samples,:]
        p3 = tdf3[rand_samples,:]
        
    
        logger = np.asarray(np.where(np.isnan(p1).astype(np.int32) == 1))[0,:]
        logger = np.append(logger, np.asarray(np.where(np.isnan(p2).astype(np.int32) == 1))[0,:])
        logger = np.append(logger, np.asarray(np.where(np.isnan(p3).astype(np.int32) == 1))[0,:])

        loggerlen = len(logger)


    for j in range(len(rand_samples)):

        logTDF.pop(int(np.asarray(np.where(np.asarray(logTDF) == rand_samples[j]))))

    return p1, p2, p3, logTDF, epoch

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

def siamese(input, reuse=False):

    '''Define network structure'''

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)


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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
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




        with tf.variable_scope('conv1', reuse=True):
            conv1_kernel = tf.get_variable('conv3d/kernel')
            conv1_kernel = tf.reshape(conv1_kernel, [1, -1])
            tf.summary.histogram('conv1_kernel', conv1_kernel)


        with tf.variable_scope('conv2', reuse=True):
            conv2_kernel = tf.get_variable('conv3d/kernel')
            conv2_kernel = tf.reshape(conv2_kernel, [1, -1])
            tf.summary.histogram('conv2_kernel', conv2_kernel)


        with tf.variable_scope('conv3', reuse=True):
            conv3_kernel = tf.get_variable('conv3d/kernel')
            conv3_kernel = tf.reshape(conv3_kernel, [1, -1])
            tf.summary.histogram('conv3_kernel', conv3_kernel)


        with tf.variable_scope('conv4', reuse=True):
            conv4_kernel = tf.get_variable('conv3d/kernel')
            conv4_kernel = tf.reshape(conv4_kernel, [1, -1])
            tf.summary.histogram('conv4_kernel', conv4_kernel)


        with tf.variable_scope('conv5', reuse=True):
            conv5_kernel = tf.get_variable('conv3d/kernel')
            conv5_kernel = tf.reshape(conv5_kernel, [1, -1])
            tf.summary.histogram('conv5_kernel', conv5_kernel)


        with tf.variable_scope('conv6', reuse=True):
            conv6_kernel = tf.get_variable('conv3d/kernel')
            conv6_kernel = tf.reshape(conv6_kernel, [1, -1])
            tf.summary.histogram('conv6_kernel', conv6_kernel)


        with tf.variable_scope('conv7', reuse=True):
            conv7_kernel = tf.get_variable('conv3d/kernel')
            conv7_kernel = tf.reshape(conv7_kernel, [1, -1])
            tf.summary.histogram('conv7_kernel', conv7_kernel)


        with tf.variable_scope('conv8', reuse=True):
            conv8_kernel = tf.get_variable('conv3d/kernel')
            conv8_kernel = tf.reshape(conv8_kernel, [1, -1])
            tf.summary.histogram('conv8_kernel', conv8_kernel)


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

def contrastive_loss(output_1, output_2, output_3):
 
    with tf.name_scope("contrastive_loss"):

        with tf.name_scope('constants'):

            margin = tf.constant(L2margin)
            one = tf.ones(
                [batch_size, 1],
                dtype=tf.float32)  
            zero = tf.constant(1e-04)
            nf =tf.constant(batch_size_train, tf.float32)

        with tf.name_scope('euclidean_distances'):

            print()
            print('***********************')
            print('loss debugging')
            print()

            distances1 = tf.abs(tf.subtract(output_1, output_2) * 100) + zero
            print()
            print('step1: subtract outputs')
            print(distances1)
            distances1 = tf.pow(distances1, 2)
            print()
            print('step2: square distance')
            print(distances1)
            distances1 = tf.reduce_sum(distances1, 1)
            print()
            print('step3: reduce sum')
            print(distances1)
            distances1 = tf.sqrt(distances1 + zero)
            print()
            print('step4: sqrt')
            print(distances1)
            distances1 = tf.reshape(distances1, [1,int(batch_size_train)])
            print()
            print('step5: reshape')
            print(distances1)

            distances2 = tf.abs(tf.subtract(output_1, output_3) * 100) + zero
            distances2 = tf.pow(distances2, 2)
            distances2 = tf.reduce_sum(distances2, 1)
            distances2 = tf.sqrt(distances2 + zero)
            distances2 = tf.reshape(distances2, [1,int(batch_size_train)])


            distances1_L = distances1[0,:int(batch_size_train/2)]
            distances1_B = distances1[0,int(batch_size_train/2):]
            distances2_L = distances2[0,:int(batch_size_train/2)]
            distances2_B = distances2[0,int(batch_size_train/2):]

            distances1_L = tf.reshape(distances1_L, [int(batch_size_train/2),1])
            distances1_B = tf.reshape(distances1_B, [int(batch_size_train/2),1])
            distances2_L = tf.reshape(distances2_L, [int(batch_size_train/2),1])
            distances2_B = tf.reshape(distances2_B, [int(batch_size_train/2),1])

            print()
            print('step6: split in branch/ leaf')
            print(distances1_L)
            print(distances1_B)
            print(distances2_L)
            print(distances2_B)


            with tf.device('/cpu:0'):

                values_matches_L = tf.contrib.framework.sort(
                    distances1_L,
                    axis=0,
                    direction='DESCENDING')[:batch_size]

                values_matches_B = tf.contrib.framework.sort(
                    distances1_B,
                    axis=0,
                    direction='DESCENDING')[:batch_size]

                values_non_matches_L = tf.contrib.framework.sort(
                    distances2_L,
                    axis=0,
                    direction='ASCENDING')[:batch_size]

                values_non_matches_B = tf.contrib.framework.sort(
                    distances2_B,
                    axis=0,
                    direction='ASCENDING')[:batch_size]


            values_matches_L = tf.reshape(values_matches_L, [batch_size,1])
            values_matches_B = tf.reshape(values_matches_B, [batch_size,1])
            values_non_matches_L = tf.reshape(values_non_matches_L, [batch_size,1])
            values_non_matches_B = tf.reshape(values_non_matches_B, [batch_size,1])


            values_matches = tf.concat([values_matches_L, values_matches_B], 0)
            values_non_matches = tf.concat([values_non_matches_L, values_non_matches_B], 0)

            print()
            print('step7: match/ nonmacth values for branch/ leaf')
            print(values_matches_L)
            print(values_matches_B)
            print(values_non_matches_L)
            print(values_non_matches_B)


            print()
            print('step8: concat values')
            print(values_matches)
            print(values_non_matches)



            dummy_non_match = tf.cast(tf.less(values_non_matches - margin, 0.), tf.float32)

            values_non_matches_over_margin = values_non_matches * dummy_non_match


            print()
            print('step9: values_non_matches_over_margin')
            print(values_non_matches_over_margin)


            # l2diff_non_match1 = tf.reduce_sum(values_non_matches_over_margin) / (tf.reduce_sum(dummy_non_match) + zero)
            l2diff_non_match1 = tf.reduce_mean(values_non_matches_over_margin)

            f1 = lambda: l2diff_non_match1
            f2 = lambda: tf.reduce_mean(values_non_matches)
            l2diff_non_match = tf.case([(tf.less_equal(l2diff_non_match1, 1), f2)], default=f1)

            l2diff_match = tf.reduce_mean(values_matches)


            l2diff_match = l2diff_match
            l2diff_non_match = l2diff_non_match



        with tf.name_scope('match_loss'):

            part1 = tf.pow(l2diff_match, 2)

        with tf.name_scope('non_match_loss'):
         
            partmax = tf.maximum(
                zero,
                tf.subtract(margin, l2diff_non_match))

            # partmax = partmax * (tf.reduce_sum(dummy_non_match) + zero) / (4*batch_size)

            part2 = tf.pow(partmax, 2)

        with tf.name_scope('regularization'):

            reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)

            reg_constant = 1
            regularization = reg_constant * sum(reg_losses)

        with tf.name_scope('final_loss'):

            # loss_non_reg = np.add(part1, part2)
            # loss = np.add(loss_non_reg, regularization)
            loss = part1 + part2 + regularization


        tf.summary.scalar('l2diff_match', tf.reshape(l2diff_match, []))
        tf.summary.scalar('l2diff_non_match', tf.reshape(l2diff_non_match, []))
        tf.summary.scalar('safety_margin', tf.subtract(l2diff_non_match, l2diff_match))
        tf.summary.scalar('match_loss', tf.reshape(part1, []))
        tf.summary.scalar('non_match_loss', tf.reshape(part2, []))
        tf.summary.scalar('loss', tf.reshape(loss, []))
        tf.summary.scalar('full_match_dist', tf.reshape(tf.reduce_mean(distances1), []))
        tf.summary.scalar('full_non_match_dist', tf.reshape(tf.reduce_mean(distances2), []))
        tf.summary.scalar('regularization', tf.reshape(tf.reduce_mean(regularization), []))
        tf.summary.histogram('match dists leaf', values_matches_L)
        tf.summary.histogram('match dists branch', values_matches_B)
        tf.summary.histogram('non match dists leaf', values_non_matches_L)
        tf.summary.histogram('non match dists brnach', values_non_matches_B
            )
        tf.summary.histogram('distances match', distances1)
        tf.summary.histogram('distances non match', distances2)

        return loss, l2diff_match, l2diff_non_match, part1, part2, distances1, distances2

def get_recon_data(idx, print_out=False):

    fx = 365.605889726
    fy = 365.605889726
    cx = 255
    cy = 211

    pixCoords1 = info1[idx].get('pixelCoords')
    pixCoords2 = info2[idx].get('pixelCoords')

    camPos1 = info1[idx].get('cam_position')
    camPos2 = info2[idx].get('cam_position')

    camCoords1 = info1[idx].get('camCoords')
    camCoords2 = info2[idx].get('camCoords')

    PointRealWorld1 = info1[idx].get('PointRealWorld')
    PointRealWorld2 = info2[idx].get('PointRealWorld')


    gamma1 = depthImg1[pixCoords1[1], pixCoords1[0]]
    gamma2 = depthImg2[pixCoords2[1], pixCoords2[0]]



    u1, v1 = pixCoords1[0], 2 * cy - pixCoords1[1]
    col1, row1 = pixCoords1[0], pixCoords1[1]

    u2, v2 = pixCoords2[0], 2 * cy - pixCoords2[1]
    col2, row2 = pixCoords2[0], pixCoords2[1]

    alpha1 = (u1 - cx) * gamma1 / fx
    alpha2 = (u2 - cx) * gamma2 / fx

    beta1 = (v1 - cy) * gamma1 / fy
    beta2 = (v2 - cy) * gamma2 / fy


    sigma1 = np.transpose(np.array([alpha1, beta1, gamma1]))
    sigma2 = np.transpose(np.array([alpha2, beta2, gamma2]))

    if print_out:
        print()
        print('pixCoords')
        print(np.round(pixCoords1, 2))
        print(np.round(pixCoords2, 2))
        print()
        print('u,v')
        print(u1, v1)
        print(u2, v2)
        print()
        print('camPos')
        print(np.round(camPos1, 2))
        print(np.round(camPos2, 2))
        print('---------translation-------> ', np.round(camPos2-camPos1, 2))
        print()
        print('camCoords')
        print(np.round(camCoords1, 2))
        print(np.round(camCoords2, 2))
        print()
        print('PointRealWorld')
        print(np.round(PointRealWorld1, 2))
        print(np.round(PointRealWorld2, 2))
        print()
        print('gamma')
        print(np.round(gamma1, 2))
        print(np.round(gamma2, 2))
        print()
        print('sigma')
        print(np.round(sigma1, 2))
        print(np.round(sigma2, 2))
        print()

    return col1, row1, col2, row2, sigma1, sigma2

def get_recon_data_all(idx):

    fx = 365.605889726
    fy = 365.605889726
    cx = 255
    cy = 211

    pixCoords1 = info1[idx].get('pixelCoords')
    pixCoords2 = info2[idx].get('pixelCoords')

    camPos1 = info1[idx].get('cam_position')
    camPos2 = info2[idx].get('cam_position')

    camCoords1 = info1[idx].get('camCoords')
    camCoords2 = info2[idx].get('camCoords')

    PointRealWorld1 = info1[idx].get('PointRealWorld')
    PointRealWorld2 = info2[idx].get('PointRealWorld')


    gamma1 = depthImg1[pixCoords1[1], pixCoords1[0]]
    gamma2 = depthImg2[pixCoords2[1], pixCoords2[0]]


    u1, v1 = pixCoords1[0], 2 * cy - pixCoords1[1]
    col1, row1 = pixCoords1[0], pixCoords1[1]

    u2, v2 = pixCoords2[0], 2 * cy - pixCoords2[1]
    col2, row2 = pixCoords2[0], pixCoords2[1]

    alpha1 = (u1 - cx) * gamma1 / fx
    alpha2 = (u2 - cx) * gamma2 / fx

    beta1 = (v1 - cy) * gamma1 / fy
    beta2 = (v2 - cy) * gamma2 / fy


    sigma1 = np.transpose(np.array([alpha1, beta1, gamma1]))
    sigma2 = np.transpose(np.array([alpha2, beta2, gamma2]))


    data = {'pixCoords1': pixCoords1,
            'pixCoords2': pixCoords2, 
            'camPos1': camPos1,
            'camPos2': camPos2,
            'camCoords1': camCoords1,
            'camCoords2': camCoords2,
            'PointRealWorld1': PointRealWorld1,
            'PointRealWorld2': PointRealWorld2,
            'u1': u1,
            'u2': u2,
            'v1': v1,
            'v2': v2,
            'col1': col1,
            'col2': col2,
            'row1': row1,
            'row2': row2,
            'alpha1': alpha1,
            'alpha2': alpha2,
            'beta1': beta1,
            'beta2': beta2,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'sigma1': sigma1,
            'sigma2': sigma2}


    return data

def squareLoss(x):

    q1, q2, x, y, z = x[:]

    quat_rec = Quaternion(q1, 0., 0., q2)

    loss = 0

    for i in range(x_data.shape[0]):

        PointRealWorld = np.array([0., 0., 0.]) + np.array([x_data[i,2], -x_data[i,0], x_data[i,1]])

        PointRealWorld2 = np.array([x, y, z]) + quat_rec.rotate(np.array([y_data[i,2], -y_data[i,0], y_data[i,1]]))

        loss += np.mean((PointRealWorld - PointRealWorld2) ** 2, 0)

    return loss/x_data.shape[0]


fx = 365.605889726
fy = 365.605889726
cx = 255
cy = 211

print()
print('load distances and meta data............')
dist = np.reshape(np.load('doublette_test9/dist1_rec_999.npy'), (-1,1))
inf1 = np.load('doublette_test9/info1_rec_999.npy')
inf2 = np.load('doublette_test9/info2_rec_999.npy')
# inf1[:,1] *= -1
# inf2[:,1] *= -1
print()


depthImg1 = convert_pfm('TreeDataJake/Cherry/Cherry_High1/00005pl.pfm')
depthImg2 = convert_pfm('TreeDataJake/Cherry/Cherry_High1/00006pl.pfm')

print('shape of distances:', dist.shape)
print('shape of info:', inf1.shape)


sigma1, sigma2 = np.zeros((dist.shape[0], 3)), np.zeros((dist.shape[0], 3))

ppp = []

plt.figure()

for i in range(dist.shape[0]):

    pixCoords1 = inf1[i,:]
    pixCoords2 = inf2[i,:]

    gamma1 = depthImg1[pixCoords1[1], pixCoords1[0]]
    gamma2 = depthImg2[pixCoords2[1], pixCoords2[0]]

    if i < 1000:
        plt.scatter(i, gamma1)
        plt.scatter(i, gamma2)

    u1, v1 = pixCoords1[0], 2 * cy - pixCoords1[1]
    col1, row1 = pixCoords1[0], pixCoords1[1]

    u2, v2 = pixCoords2[0], 2 * cy - pixCoords2[1]
    col2, row2 = pixCoords2[0], pixCoords2[1]

    alpha1 = (u1 - cx) * gamma1 / fx
    alpha2 = (u2 - cx) * gamma2 / fx

    beta1 = (v1 - cy) * gamma1 / fy
    beta2 = (v2 - cy) * gamma2 / fy


    sigma1[i,:] = np.reshape(np.array([gamma1, -alpha1, beta1]), (1,3))
    sigma2[i,:] = np.reshape(np.array([gamma2, -alpha2, beta2]), (1,3))
plt.grid()


plt.figure()
plt.scatter(inf2[:,0], inf2[:,1])


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot(testvec2[0,:], testvec2[1,:], testvec2[2,:], c='r', label='sight2')

# ax.scatter(sigma1[np.mean(sigma1, 1) != 0,0], sigma1[np.mean(sigma1, 1) != 0,1], sigma1[np.mean(sigma1, 1) != 0,2], c='b', s=4, label='pco1')
# ax.scatter(sigma2[np.mean(sigma1, 1) != 0,0], sigma2[np.mean(sigma1, 1) != 0,1], sigma2[np.mean(sigma1, 1) != 0,2], c='g', s=4, label='pco2')

# ax.legend()
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


plt.figure()
plt.plot(np.sort(sigma1[:,0]))
plt.plot(np.sort(sigma1[:,1]))
plt.plot(np.sort(sigma1[:,2]))
plt.plot(np.sort(sigma2[:,0]))
plt.plot(np.sort(sigma2[:,1]))
plt.plot(np.sort(sigma2[:,2]))
plt.grid()



plt.figure()
plt.imshow(depthImg1)
plt.scatter(inf1[:10000,0], inf1[:10000,1], s=2)


plt.figure()
plt.imshow(depthImg2)
plt.scatter(inf2[:10000,0], inf2[:10000,1], s=2)



# rank_threshold = 100
# loss_threshold = 10

plots = True



# arbitrary_point = np.reshape(np.zeros(3), (3))

# point_cloud1, point_cloud2 = [], []


# for i in range(len(d1idx)):

#     idx = d1idx[i]

#     s1 = get_recon_data_all(idx).get('sigma1', np.zeros(3))
#     s2 = get_recon_data_all(idx).get('sigma2', np.zeros(3))

#     if np.mean(np.abs(s1)) != 0 and np.mean(np.abs(s2)) != 0:

#         sig1 = np.array([s1[2], -s1[0], s1[1]])
#         sig2 = np.array([s2[2], -s2[0], s2[1]])

#         point_cloud1.append(arbitrary_point + sig1)
#         point_cloud2.append(arbitrary_point + sig2)


# point_cloud1 = np.asarray(point_cloud1)
# point_cloud2 = np.asarray(point_cloud2)


# centroid1 = np.mean(point_cloud1, 0)
# centroid2 = np.mean(point_cloud2, 0)

# pco1 = (point_cloud1 - centroid1)# / np.linalg.norm(point_cloud1 - centroid1)
# pco2 = (point_cloud2 - centroid2)# / np.linalg.norm(point_cloud2 - centroid2)



# ransacLog = np.arange(len(d1idx))


# def comp_rot_error(data1, data2):

#     if data1.shape[1] != 3:
#         data1 = np.reshape(data1, (3,-1))
#         data2 = np.reshape(data2, (3,-1))
#         print('watch out for dimensions in rot error comp')

#     centroid1 = np.mean(data1, 0)
#     centroid2 = np.mean(data2, 0)

#     cloud1 = (data1 - centroid1)# / np.linalg.norm(point_cloud1 - centroid1)
#     cloud2 = (data2 - centroid2)# / np.linalg.norm(point_cloud2 - centroid2)


#     H = np.zeros((3,3))

#     for i in range(cloud1.shape[0]):

#         H += np.dot(np.reshape(cloud1[i,:], (3,1)), np.reshape(cloud2[i,:], (1,3)))


#     U, S, V = np.linalg.svd(H)

#     R = np.dot(V, np.transpose(U))


#     if np.linalg.det(R) < 0:

#         V[:,2] *= -1

#         R = np.dot(V, np.transpose(U))

#     T = np.dot(-R, centroid1) + centroid2
#     T = np.reshape(T, (3,1))

#     new_pc2 = np.dot(R, np.transpose(point_cloud1)) + T

#     diff = np.sum((np.reshape(point_cloud2, (3, -1)) - new_pc2) ** 2, 0)
#     diff_scalar = np.mean(diff)

#     return diff, diff_scalar, R, T, U, S, V

# def ransac(point_cloud1, point_cloud2, ransacLog):

#     err0 = 100
#     needed_points = 6
#     wanted_points = 20
#     ransac_iterations = 10000

#     error_prog = []

#     trusted_ones = []


#     for i in range(ransac_iterations):

#         candidates = np.random.choice(len(ransacLog), needed_points, replace=False)

#         cloud1, cloud2 = point_cloud1[candidates,:], point_cloud2[candidates,:]

#         error, error_scalar, _, _, _, _, _ = comp_rot_error(cloud1, cloud2)

#         if error_scalar <= err0:
#             err0 = error_scalar
#             error_prog.append(error_scalar)
#             print(error_scalar)

#         if i == 0:

#             trusted_ones = np.asarray(ransacLog[np.where(error <= loss_threshold)])

#         else:

#             trusted_ones = np.append(trusted_ones, np.asarray(ransacLog[np.where(error <= loss_threshold)]))

#         # plt.plot(np.transpose(error))

#     print(np.asarray(trusted_ones).shape)


#     trustlog = np.zeros(len(ransacLog))
#     for i in range(len(ransacLog)):

#         trustlog[i] = np.asarray(np.where(trusted_ones == i)).shape[1]


#     usepoints = np.reshape(np.asarray(np.where(trustlog >= np.sort(trustlog)[-wanted_points])), (-1))

#     print('----------', np.asarray(trustlog[usepoints], dtype=np.int32))
#     print('----------', usepoints.shape)

#     cloud1, cloud2 = point_cloud1[np.asarray(usepoints, dtype=np.int32),:], point_cloud2[np.asarray(usepoints, dtype=np.int32),:]
#     print('----------', cloud1.shape)

#     error, error_scalar, ransac_rot, trans, u, s, v = comp_rot_error(cloud1, cloud2)





#     print()
#     print('---------------------------------------')
#     print('ransac:')
#     print()
#     print('final error:', error_scalar)
#     print()
#     print('rotation:')
#     print(ransac_rot)
#     print()
#     print('U')
#     print(u)
#     print()
#     print('S')
#     print(s)
#     print()
#     print('V')
#     print(v)
#     print('---------------------------------------')
#     print()


#     plt.figure()
#     plt.plot(error_prog)
#     plt.title('error_prog')
#     plt.grid()


#     plt.figure()

#     plt.scatter(np.arange(len
#         (trustlog)), trustlog, c='b')
#     plt.scatter(usepoints, trustlog[usepoints], c='r')
#     plt.grid()


#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.scatter(pco1[:,0], pco1[:,1], pco1[:,2], c='b', marker='o', label='pco1')
#     ax.scatter(pco2[:,0], pco2[:,1], pco2[:,2], c='g', marker='o', label='pco2')

#     zrot = np.array([[np.cos(.35), -np.sin(.35), 0.], [np.sin(.35), np.cos(.35), 0.], [0., 0., 1.]])
 
#     new_pc2 = np.dot(zrot, np.transpose(pco1))

#     ax.scatter(new_pc2[0,:], new_pc2[1,:], new_pc2[2,:], c='r', marker='x', label='pco1_rot')

#     # for i in range(pco1.shape[0]):
#     #     ax.plot(np.linspace(pco2[i,0], pco1rot[i,0]), np.linspace(pco2[i,1], pco1rot[i,1]), np.linspace(pco2[i,2], pco1rot[i,2]), c='m')

#     ax.legend()
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')


# ransac(point_cloud1, point_cloud2, ransacLog)




if plots:

    # best_col1, best_row1, best_col2, best_row2, _, _ = get_recon_data(bestidx)

    # plt.figure()
    # plt.plot(np.transpose(np.sort(d1)))
    # plt.title('distances')
    # plt.grid()


    # testimage1 = imageio.imread(DataPath + 'Cherry' + '/' + 'Cherry' + '_High' + str(1) + '/' + file_index(5) + '.ppm')
    # testimage2 = imageio.imread(DataPath + 'Cherry' + '/' + 'Cherry' + '_High' + str(1) + '/' + file_index(6) + '.ppm')

    # plt.figure(figsize=(10,10))
    # plt.subplot(221)
    # plt.imshow(depthImg1)
    # # plt.scatter(rand_col, rand_row, c="r")

    # for i in range(len(d1idx2nd)):
    #     fig_col1, fig_row1, fig_col2, fig_row2, _, _ = get_recon_data(d1idx2nd[i])
    #     plt.scatter(fig_col1, fig_row1, 8, 'b')
    # fig_col1, fig_row1, fig_col2, fig_row2, _, _ = get_recon_data(d1idx2nd[-1])
    # plt.scatter(fig_col1, fig_row1, 8, 'b', label='<'+str(2*rank_threshold))

    # for i in range(len(d1idx)):
    #     fig_col1, fig_row1, fig_col2, fig_row2, _, _ = get_recon_data(d1idx[i])
    #     plt.scatter(fig_col1, fig_row1, 8, 'm')
    
    # plt.scatter(fig_col1, fig_row1, 8, 'm', label='<'+str(rank_threshold))

    # plt.scatter(best_col1, best_row1, 10, 'r', label='best match')

    # # plt.scatter(np.array(np.nonzero(depthImg1))[1,:],np.array(np.nonzero(depthImg1))[0,:])
    # plt.grid()
    # plt.legend()

    # plt.subplot(222)
    # plt.imshow(depthImg2)
    # plt.title('corresponding view point')

    # for i in range(len(d1idx2nd)):
    #     fig_col1, fig_row1, fig_col2, fig_row2, _, _ = get_recon_data(d1idx2nd[i])
    #     plt.scatter(fig_col2, fig_row2, 8, 'b')

    # for i in range(len(d1idx)):
    #     fig_col1, fig_row1, fig_col2, fig_row2, _, _ = get_recon_data(d1idx[i])
    #     plt.scatter(fig_col2, fig_row2, 8, 'm')

    # plt.scatter(best_col2, best_row2, 10, 'r', label='best match')

    # # plt.scatter(u2, v2, 80, 'y', 'x')
    # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
    # # plt.scatter(pixX, pixY, 80, 'r', 'x')
    # plt.grid()
    # plt.legend()

    # plt.subplot(223)
    # plt.imshow(testimage1)
    # plt.title('image 1')
    # plt.scatter(best_col1, best_row1, 10, 'r', label='best match')

    # # for i in range(len(d1idx2nd)):
    # #     fig_col1, fig_row1, fig_col2, fig_row2 = get_recon_data(d1idx2nd[i])
    # #     plt.scatter(fig_row1, fig_col1, 10, 'b')

    # # for i in range(len(d1idx)):
    # #     fig_col1, fig_row1, fig_col2, fig_row2 = get_recon_data(d1idx[i])
    # #     plt.scatter(fig_row1, fig_col1, 10, 'm')

    # plt.grid()
    # plt.legend()

    # plt.subplot(224)
    # plt.imshow(testimage2)
    # plt.title('image 2')
    # plt.scatter(best_col2, best_row2, 10, 'r', label='best match')

    # # for i in range(len(d1idx2nd)):
    # #     fig_col1, fig_row1, fig_col2, fig_row2 = get_recon_data(d1idx2nd[i])
    # #     plt.scatter(fig_row2, fig_col2, 10, 'b')

    # # for i in range(len(d1idx)):
    # #     fig_col1, fig_row1, fig_col2, fig_row2 = get_recon_data(d1idx[i])
    # #     plt.scatter(fig_row2, fig_col2, 10, 'm')

    # # # plt.scatter(u2, v2, 80, 'y', 'x')
    # # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
    # # # plt.scatter(pixX, pixY, 80, 'r', 'x')
    # plt.grid()
    # plt.legend()



    # pt = np.dot(R, np.ones((3,1)))
    # testvec1 = np.reshape(np.array([[np.linspace(0, 1, 50)], [np.linspace(0, 1, 50)], [np.linspace(0, 1, 50)]]), (3,-1))
    # # testvec2 = np.reshape(np.array([[np.linspace(0, pt[0,0], 50)], [np.linspace(0, pt[1,0], 50)], [np.linspace(0, pt[2,0], 50)]]), (3,-1))
    # testvec2 = np.dot(R, testvec1)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # ax.plot(testvec1[0,:], testvec1[1,:], testvec1[2,:], c='b', label='sight1')
    # # ax.plot(testvec2[0,:], testvec2[1,:], testvec2[2,:], c='r', label='sight2')

    # ax.scatter(pco1[:,0], pco1[:,1], pco1[:,2], c='b', label='pco1')
    # ax.scatter(pco2[:,0], pco2[:,1], pco2[:,2], c='g', label='pco2')
    # pco1rot = np.transpose(np.dot(R, np.transpose(pco2)))
    # ax.scatter(pco1rot[:,0], pco1rot[:,1], pco1rot[:,2], c='r', label='pco2_rot')

    # ax.legend()
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')




   
    plt.show()








