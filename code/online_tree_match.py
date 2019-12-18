from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# from tensorflow.python.util.tf_export import tf_export
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


## Network Setting
#########################
#########################
DATA_DIR = 'tf_logs3/'

iterations = 1000000
batch_size = 1
batch_size_train = 3

window_size = 30

L2margin = 80.

test_size = 10
acc_print_freq = 1000
moving_avg_window = 100

test_multiplier = 100

what = 'quick_frame30'
zero = tf.constant(1e-04)
#########################
#########################

## Data Loader Settings
#########################
#########################
DataPath = 'TreeDataJake/'
trees = np.asarray(['Cherry', 'KoreanStewartia'])
num_tree_folders = 50
num_tree_pics = np.asarray([789, 789, 709, 429])

random_tree_debug = 48
snapshot1_index_debug = 3
rand_row_debug, rand_col_debug = 117, 252
snapshot2_index_debug = 2

voxelGridPatchRadius = 15  # in voxels
voxelSize = 0.01  # in meters
voxelMargin = voxelSize * 5
#########################
#########################

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


def getPatchData_gpu(pointData,voxelGridPatchRadius,voxelSize,voxelMargin):

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
    gamma_tolerance = .2
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


    return p1_voxelGridTDF, p2_voxelGridTDF, p3_voxelGridTDF, point_label1

####################################################

def TDF_train_loader_no_labels(batchsize):

    counter = 0

    tdf1 = np.zeros((batchsize, window_size, window_size, window_size, 1))
    tdf2 = np.zeros((batchsize, window_size, window_size, window_size, 1))
    tdf3 = np.zeros((batchsize, window_size, window_size, window_size, 1))

    error_solution = np.zeros((batch_size_train, window_size, window_size, window_size, 1))

    while counter < batchsize:

        p1, p2, p3, label1 = doublette('Cherry')

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
    
    # print('data assembly successful')


    return tdf1, tdf2, tdf3

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

        tf.summary.histogram('output_conv8', conv8)

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
            zero = tf.constant(1e-06)
            nf =tf.constant(batch_size_train, tf.float32)

        with tf.name_scope('euclidean_distances'):

            distances1 = tf.subtract(output_1, output_2)
            distances1 = tf.pow(distances1, 2)
            distances1 = tf.reduce_mean(distances1, 1)
            distances1 = tf.sqrt(distances1)
            distances1 = tf.reshape(distances1, [-1,1]) * 1e05

            distances2 = tf.subtract(output_1, output_3)
            distances2 = tf.pow(distances2, 2)
            distances2 = tf.reduce_mean(distances2, 1)
            distances2 = tf.sqrt(distances2)
            distances2 = tf.reshape(distances2, [-1,1]) * 1e05

            distances1_L = distances1[:int(batch_size_train/2)]
            distances1_B = distances1[int(batch_size_train/2):]
            distances2_L = distances2[:int(batch_size_train/2)]
            distances2_B = distances2[int(batch_size_train/2):]


            values_matches_L = tf.contrib.framework.sort(
                distances1_L,
                axis=-1,
                direction='DESCENDING')[:batch_size]

            values_matches_B = tf.contrib.framework.sort(
                distances1_B,
                axis=-1,
                direction='DESCENDING')[:batch_size]

            values_non_matches_L = tf.contrib.framework.sort(
                distances2_L,
                axis=-1,
                direction='ASCENDING')[:batch_size]

            values_non_matches_B = tf.contrib.framework.sort(
                distances2_B,
                axis=-1,
                direction='ASCENDING')[:batch_size]

            values_matches = tf.concat([values_matches_L, values_matches_B], 0)
            values_non_matches = tf.concat([values_non_matches_L, values_non_matches_B], 0)



            dummy_non_match = tf.cast(tf.less(values_non_matches - margin, 0.), tf.float32)

            values_non_matches_over_margin = values_non_matches * dummy_non_match

            l2diff_non_match1 = tf.reduce_sum(values_non_matches_over_margin) / (tf.reduce_sum(dummy_non_match) + zero)

            f1 = lambda: l2diff_non_match1
            f2 = lambda: tf.reduce_mean(values_non_matches)
            l2diff_non_match = tf.case([(tf.equal(l2diff_non_match1, 0.), f2)], default=f1)

            l2diff_match = tf.reduce_mean(values_matches)





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

            reg_constant = 1e-10
            regularization = reg_constant * sum(reg_losses)

        with tf.name_scope('final_loss'):

            loss_non_reg = np.add(part1, part2)
            # loss = np.add(loss_non_reg, regularization)
            loss = loss_non_reg

        # with tf.name_scope('monitor'):

            # accuracy, error_rate, recall, true_matches_sum, false_matches_sum, true_non_matches_sum, false_non_matches_sum = stats(l2distance, margin, labels, 'full')

            # accuracy_trunk, accuracy_branches, accuracy_leaves, error_rate_trunk, error_rate_branches, error_rate_leaves, recall_trunk, recall_branches, recall_leaves = stats(l2distance, margin, labels, 'split')

            # accuracy_trunk_dyn, accuracy_branches_dyn, accuracy_leaves_dyn, accuracy_test_dyn, error_rate_trunk_dyn, error_rate_branches_dyn, error_rate_leaves_dyn, error_rate_test_dyn = stats(l2distance, threshold, labels, 'split')


        tf.summary.scalar('l2diff_match', tf.reshape(l2diff_match, []))
        tf.summary.scalar('l2diff_non_match', tf.reshape(l2diff_non_match, []))
        tf.summary.scalar('safety_margin', tf.subtract(l2diff_non_match, l2diff_match))
        tf.summary.scalar('match_loss', tf.reshape(part1, []))
        tf.summary.scalar('non_match_loss', tf.reshape(part2, []))
        tf.summary.scalar('loss', tf.reshape(loss, []))
        tf.summary.scalar('full_match_dist', tf.reshape(tf.reduce_mean(distances1), []))
        tf.summary.scalar('full_non_match_dist', tf.reshape(tf.reduce_mean(distances2), []))
        tf.summary.histogram('distances match', distances1)
        tf.summary.histogram('distances non match', distances2)

        return loss, l2diff_match, l2diff_non_match, part1, part2, distances1, distances2

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

x3 = tf.placeholder(
    tf.float32, 
    [batch_size_train, window_size, window_size, window_size, 1],
    name="input3")


output1 = siamese(x1, reuse=False)
output2 = siamese(x2, reuse=True)
output3 = siamese(x3, reuse=True)


loss, l2diff_match, l2diff_non_match, match_loss, non_match_loss, distances1, distances2 = contrastive_loss(output1, output2, output3)


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


def moving_threshold(threshold_history, i):

    if i == 0:

        return L2margin
    elif i < 50:

        return np.mean(threshold_history[:i])

    elif i >= 50:

        return np.mean(threshold_history[i - moving_avg_window:i])



with tf.Session() as sess:

    # saver = tf.train.import_meta_graph('tensorflow_logs/doublette_test5/model.ckpt-9999.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('tensorflow_logs/doublette_test5/'))

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
            name='AdamOpt'
            ).minimize(loss)

    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(
        DATA_DIR,
        graph = sess.graph)




    for i in range(0,iterations):


        data1, data2, data3 = TDF_train_loader(batch_size_train)


        summary, _ , loss_train, l2md, l2nonmd, ML, NML = sess.run([
            merged, 
            train_step, 
            loss, 
            l2diff_match, 
            l2diff_non_match,
            match_loss,
            non_match_loss
            ], 
            feed_dict={
                x1: data1,
                x2: data2,
                x3: data3
                })


        train_writer.add_summary(summary, i)
        saver.save(sess, os.path.join(DATA_DIR, "model.ckpt"), i)


        if i % 1 == 0:
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print('########################################################################### iteration ', i)
            print('@', DATA_DIR, 'margin->', L2margin, ' | batch size->', batch_size_train)
            print('worst', batch_size, ' samples used to train')
            print('###########################################################################')
            print('loss...........................................', loss_train)
            print('l2diff_match...................................', l2md)
            print('l2diff_non_match...............................', l2nonmd)
            print('match_loss.....................................', ML)
            print('non_match_loss.................................', NML)
            # print(test_dist)
            # print(trunk3)
            print()
            print()
            print()
            print()
            print()
            print()
            print()


        if i % acc_print_freq == 0 and i > 0:

            for h in range(test_multiplier):

                if h % 2 == 0:

                    progress = h/test_multiplier*100
                    progress = np.round(progress)

                    print(progress, '%', 'of sampling done.')

                if h == 0:

                    storage_match_distances = []
                    storage_non_match_distances = []

                data1_test, data2_test, data3_test = TDF_train_loader(batch_size_train)
                
                distances1_test, distances2_test = sess.run([distances1, distances2], feed_dict={
                x1: data1_test,
                x2: data2_test,
                x3: data3_test})

                storage_match_distances.append(distances1_test)
                storage_non_match_distances.append(distances2_test)

            storage_match_distances = np.asarray(storage_match_distances)
            storage_non_match_distances = np.asarray(storage_non_match_distances)

            runData = np.append(storage_match_distances, storage_non_match_distances)

            filename_dists = DATA_DIR + 'runData_' + str(i) + '.npy'
            print('saving distances to ', filename_dists)
            np.save(filename_dists, runData)

            runData = []



        if np.isnan(loss_train):
            print('Model diverged with loss = NaN')
            quit()


# print(w_1)



