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
import h5py
from tensorflow.contrib.tensorboard.plugins import projector
from numba import vectorize, jit

## Network Setting
#########################
#########################
DATA_DIR = 'tensorflow_logs/doublette_test6/'

iterations = 1000000
batch_size = 4
batch_size_train = 8

window_size = 30

L2margin = 80.

test_size = 10
acc_print_freq = 500
moving_avg_window = 100

test_multiplier = 25

what = 'quick_frame30'
zero = tf.constant(1e-04)
#########################
#########################

## Data Loader Settings
#########################
#########################
# DataPath = 'TreeDataJake/'
DataPath = '/media/drzadmin/DATA/TreeDataJake/'
trees = np.asarray(['Cherry', 'KoreanStewartia'])
num_tree_folders = 100
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

@jit
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

        print(camPts)

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


        n1 = torch.FloatTensor(camPts + 1e-06).cuda()


        part = int(np.round(gridPtsCam.shape[0]/3))

        n2_1 = torch.FloatTensor(gridPtsCam[:part,:] + 1e-06).cuda()
        n2_2 = torch.FloatTensor(gridPtsCam[part:2*part,:] + 1e-06).cuda()
        n2_3 = torch.FloatTensor(gridPtsCam[2*part:,:] + 1e-06).cuda()
        #print(n1.size())
        #knnDist = torch.FloatTensor(n2.size()[0],1).cuda()
        #for i in range(n2.size()[0]):
        #    dist = torch.sum((n1 - n2[i,:])**2,1)
        #    knnDist[i] = torch.min(dist)
        #    knnDist[i] = torch.sqrt(knnDist[i])

        #sum_1 = torch.t(torch.sum(n1**2, 1).repeat(n2.size()[0],1))
        #sum_2 = torch.sum(n2**2, 1).repeat(n1.size()[0],1)
        knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_1.size()[0],1))+torch.sum(n2_1**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_1)),0)
        knnDist = torch.clamp(knnDist,min=0.0)
        knnDist1 = torch.sqrt(knnDist)

        knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_2.size()[0],1))+torch.sum(n2_2**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_2)),0)
        knnDist = torch.clamp(knnDist,min=0.0)
        knnDist2 = torch.sqrt(knnDist)

        knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_3.size()[0],1))+torch.sum(n2_3**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_3)),0)
        knnDist = torch.clamp(knnDist,min=0.0)
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

    # segmentation2 = imageio.imread(random_tree_path + file_index2 + 'seg.ppm')
    # # segmentation2 = np.round(np.mean(segmentation2, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation2,2), (-1)))), 3)
    # segmentation2 = np.round(np.sum(segmentation2, 2), 3)


    '''label point'''
    point_label1 = segmentation1[rand_row, rand_col]

    point_label2 = 0 #segmentation2[row2, col2]



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


                random_spot3 = np.random.randint(0, np.array(np.nonzero(depthImg3)).shape[1])
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
    # segmentation3 = imageio.imread(random_tree_path + file_index3 + 'seg.ppm')
    # # segmentation3 = np.round(np.mean(segmentation3, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation3,2), (-1)))), 1)
    # segmentation3 = np.round(np.sum(segmentation3, 2), 3)



    p3_bboxCorners_test = []
    for j in range(8):
        p3_bboxCorners_test.append(lens3position + quaternion_cam3.rotate(np.array([p3_bboxCornersCam[2,j], -p3_bboxCornersCam[0,j], p3_bboxCornersCam[1,j]])))
    p3_bboxCorners_test = np.asarray(p3_bboxCorners_test)



    '''label point'''
    point_label3 = 0 #segmentation3[row3, col3]
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

@jit
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

        print()
        print(counter_leaves + counter_non_leaves, ' of ' + str(batchsize) + ' loaded')
        print()

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

    print('shapes:')
    print(tdf1.shape)
    print(tdf2.shape)
    print(tdf3.shape)
    

    return np.asarray(tdf1), np.asarray(tdf2), np.asarray(tdf3)





























for i in range(10):


    tdf1, tdf2, tdf3 = TDF_train_loader(2000)

    print()
    print('batch number ' + str(i) + ' loaded')
    print(tdf1.shape)

    np.save('tdf1_new' + str(1) + '.npy', tdf1)
    np.save('tdf2_new' + str(1) + '.npy', tdf2)
    np.save('tdf3_new' + str(1) + '.npy', tdf3)

    tdf1, tdf2, tdf3 = [], [], [] 

    print()
    print('batch number ' + str(i) + ' written')

