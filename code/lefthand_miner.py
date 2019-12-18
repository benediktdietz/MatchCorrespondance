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

DataPath = 'TreeDataJake/'

trees = np.asarray(['Cherry', 'KoreanStewartia'])
num_tree_folders = 100
num_tree_pics = np.asarray([789, 789, 709, 429])

plots = True

random_tree_debug = 20
snapshot1_index_debug = 5
rand_row_debug, rand_col_debug = 98, 375
snapshot2_index_debug = 22


voxelGridPatchRadius = 15  # in voxels
voxelSize = 0.01  # in meters
voxelMargin = voxelSize * 5


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

def doublette(whichTrees, plots=True, output=True):
    
    ############################
    depth_limit = 8.
    cam_distance_threshold = 4.
    frame_size = 30

    match_found = 0
    non_match_found = 0
    first_img_found = 0
    ############################

    fx, fy, cx, cy = get_cam_intrinsics()
    drone_cam_translation = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., .46], [0., 0., 0., 1.]])
    print('fx, fy, cx, cy...............', fx, fy, cx, cy)


    while (match_found < 1 and non_match_found < 1):

        print()
        print('finding initial camera position with visible tree..........................')
        print()

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
                random_tree = np.random.randint(num_tree_folders)
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            if whichTrees == 'KoreanStewartia':

                random_tree_kind = 1
                random_tree_kind_str = trees[random_tree_kind]
                random_tree = np.random.randint(num_tree_folders)
                random_tree_path = DataPath + random_tree_kind_str + '/' + random_tree_kind_str + '_High' + str(random_tree) + '/'

            '''find total number of frames in respective folder'''
            data_table = np.asarray(pd.read_table(random_tree_path + 'poses.txt' , sep='\s', header=0, index_col=False, engine='python'))
            num_frames = np.max(data_table[:,0])

            abc = np.squeeze(np.array(np.where(data_table[:,7] == 1)))
            mean_x = np.mean(data_table[abc[0]:abc[-1],1])
            mean_y = np.mean(data_table[abc[0]:abc[-1],2])

            # drone_trajectory_x = (data_table[:,1] - mean_x)
            # drone_trajectory_y = (data_table[:,2] - mean_y)
            # drone_trajectory_z = data_table[:,3] * -1

            drone_trajectory_z = data_table[:,3] * -1
            drone_trajectory_x = (data_table[:,1] - mean_x)
            drone_trajectory_y = (data_table[:,2] - mean_y)

            drone_trajectory = np.array([drone_trajectory_x, drone_trajectory_y, drone_trajectory_z])
            # drone_trajectory = np.dot(build_rot_mat(-.5*np.pi, 'y'), drone_trajectory)

            drone_trajectory_x = drone_trajectory[0, :]
            drone_trajectory_y = drone_trajectory[1, :]
            drone_trajectory_z = drone_trajectory[2, :]


            '''find random initial frame'''
            snapshot1_index = np.random.randint(1, num_frames)
            if whichTrees == 'debug':
                snapshot1_index = snapshot1_index_debug

            '''load respective depth img'''
            file_index1 = file_index(snapshot1_index)


            depthImg1 = convert_pfm(random_tree_path + file_index1 + 'pl.pfm')
            '''delete too far away points'''
            depthImg1[depthImg1 > depth_limit] = 0 


            '''check if there is a tree'''
            if np.array(np.nonzero(depthImg1[frame_size:-frame_size,frame_size:-frame_size])).shape[1] > 10:
                first_img_found = 1
                if output:
                    print('num_frames...............', num_frames)
                    print('random_tree_kind.........', random_tree_kind)
                    print('random_tree_kind_str.....', random_tree_kind_str)
                    print('random_tree..............', random_tree)
                    print('random_tree_path.........', random_tree_path)
                    print('index1...................', snapshot1_index)
                    print(random_tree_path + file_index1 + 'pl.pfm')
                    print()
                    print('=======================================================> initial frame found')
                    print()


        '''load the segmentation'''
        segmentation1 = imageio.imread(random_tree_path + file_index1 + 'seg.ppm')
        segmentation1 = np.round(np.mean(segmentation1, 2) / np.max(np.asarray(np.reshape(np.mean(segmentation1,2), (-1)))), 1)


        '''delete depth image parts without tree'''
        # depthImg1[.7 < segmentation1 and segmentation1 < .9] = 0

        '''choose random point on tree'''
        rand_row, rand_col = 0, 0

        while np.absolute(cy - rand_row) > (cy - frame_size) or np.absolute(cx - rand_col) > (cx - frame_size):
            random_spot1 = np.random.randint(0, np.array(np.nonzero(depthImg1)).shape[1])
            rand_row, rand_col = np.array(np.nonzero(depthImg1))[:,random_spot1]

        if whichTrees == 'debug':
            rand_row, rand_col = rand_row_debug, rand_col_debug      


        '''label point'''
        point_label = segmentation1[rand_row, rand_col]


        '''get 1st cam position and quaternions'''
        cam_position1 = get_cam_position(snapshot1_index, random_tree_path)
        qx1 = cam_position1[3]
        qy1 = cam_position1[4]
        qz1 = cam_position1[5]
        if qz1 == 0.:
            qz1 = 1e-04
        qw1 = cam_position1[6]

        quaternion_cam1 = Quaternion(qw1, qx1, qy1, qz1)

        angle1 = -2 * np.arccos(cam_position1[-1])

        x1, y1, z1 = cam_position1[:3]
        x1 = x1 - mean_x
        y1 = y1 - mean_y
        z1 = -1 * z1
        drone1position = np.transpose(np.array([x1, y1, z1]))
        lens1position = drone1position + quaternion_cam1.rotate(np.array([drone_cam_translation[2,3], 0, 0]))


        if output:
            print('rand_row, rand_col.......', rand_row, rand_col)
            print('label 1st point..........', point_label)
            print('x1, y1, z1..........', np.round(x1, 2), np.round(y1, 2), np.round(z1, 2))
            print()
            print('=======================================================> 1st frame fully loaded')
            print()


        '''find 2nd camera position in proximity'''
        cam_distance = 100.
        loop_cam2_search = 0

        while cam_distance > cam_distance_threshold:

            '''choose random and different 2nd point'''
            snapshot2_index = snapshot1_index
            while snapshot2_index == snapshot1_index:
                snapshot2_index = np.random.randint(1, num_frames)
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

            angle2 = -2 * np.arccos(cam_position2[-1])

            rotmat_z2 = build_rot_mat(angle2, 'z')
      
            x2, y2, z2 = cam_position2[:3]
            x2 = x2 - mean_x
            y2 = y2 - mean_y
            z2 = -1 * z2
            drone2position = np.transpose(np.array([x2, y2, z2]))

            lens2position = drone2position + quaternion_cam2.rotate(np.array([drone_cam_translation[2,3], 0, 0]))



            '''check if relative distance under threshold'''
            cam_distance = dist_calc(np.array([x1, y1, z1]), np.array([x2, y2, z2]))


        if output:
            print('snapshot2_index..........', snapshot2_index)
            print('x2, y2, z2...............', np.round(x2, 2), np.round(y2, 2), np.round(z2, 2))
            print('cam_distance.............', np.round(cam_distance, 2))
            print(random_tree_path + file_index2 + 'pl.pfm')
            print()
            print('=======================================================> 2nd frame found')
            print()


        '''Compute transformtation'''
        angle_delta = angle2 - angle1
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

        PointRealWorld = lens1position + quaternion_cam1.rotate(np.array([gamma1, -alpha1, beta1]))

        lens2point1 = PointRealWorld - lens1position
        lens2point2 = PointRealWorld - lens2position


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
        gamma2 = newSigma[0]

        sigma2 = np.transpose(np.array([alpha2, beta2, gamma2, 1.]))

        PointRealWorld2 = lens2position + quaternion_cam2.rotate(np.array([gamma2, -alpha2, beta2]))


        d2 = gamma2

        u2 = (alpha2 * fx / d2) + cx
        v2 = (beta2 * fy / d2) + cy

        col2 = u2
        row2 = 2 * cy - v2






        print('######################################################')
        print(sigma2)
        print('######################################################')
        print('quaternion_cam2.inverse.rotate(lens2point2)')
        print()
        print(quaternion_cam2.inverse.rotate(lens2point2))
        print('######################################################')
        print(PointRealWorld)
        print(PointRealWorld2)
        print('######################################################')
        print('######################################################')



        if output:
            print('translation..............', np.round(translation, 2))
            print('angle1...................', np.round(angle1, 2))
            print('angle2...................', np.round(angle2, 2))
            print('angle delta..............', np.round(angle_delta, 2))
            print()
            print('u1, v1...................', u1, v1)
            print('sigma1...................', np.round(sigma1, 2))

            print()

        if plots:

            '''load respective depth img'''
            depthImg2 = convert_pfm(random_tree_path + file_index2 + 'pl.pfm')
            depthImg2[depthImg2 > depth_limit] = 0



            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x1, y1, z1, c='r', label='cam_position1')
            ax.scatter(lens1position[0], lens1position[1], lens1position[2], c='r', label='lens1')
            ax.scatter(x2, y2, z2, c='b', label='cam_position2')
            ax.scatter(lens2position[0], lens2position[1], lens2position[2], c='b', label='lens2')
            ax.scatter(PointRealWorld[0], PointRealWorld[1], PointRealWorld[2], c='g', label='PointRealWorld')
            # ax.scatter(bboxCorners[0,:] + real_point[0], bboxCorners[1,:] + real_point[1], bboxCorners[2,:] + real_point[2], c='g', label='corners')
            ax.plot(drone_trajectory_x, drone_trajectory_y, drone_trajectory_z, c='y', label='drone flight')
            ax.plot(lens2point1_vector[0], lens2point1_vector[1], lens2point1_vector[2], c='r', label='sight1')
            ax.plot(lens2point2_vector[0], lens2point2_vector[1], lens2point2_vector[2], c='b', label='sight2')
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

            plt.figure(figsize=(10,10))
            plt.subplot(221)
            plt.imshow(depthImg1)
            # plt.scatter(rand_col, rand_row, c="r")
            plt.scatter(rand_col, rand_row, 80, 'w', 'x')
            #plt.scatter(np.array(np.nonzero(depthImg1))[1,:],np.array(np.nonzero(depthImg1))[0,:])
            plt.title('considered point:' + str(point_label))
            plt.grid()

            plt.subplot(222)
            plt.imshow(depthImg2)
            plt.title('corresponding view point')
            plt.scatter(col2, row2, 80, 'w', 'x')
            # # plt.scatter(u2, v2, 80, 'y', 'x')
            # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
            # # plt.scatter(pixX, pixY, 80, 'r', 'x')
            plt.grid()

            plt.subplot(223)
            plt.imshow(testimage1)
            plt.title('image 1')
            plt.scatter(rand_col, rand_row, 80, 'r', 'x')
            plt.grid()

            plt.subplot(224)
            plt.imshow(testimage2)
            plt.title('image 2')
            plt.scatter(col2, row2, 80, 'r', 'x')
            # # plt.scatter(u2, v2, 80, 'y', 'x')
            # # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
            # # plt.scatter(pixX, pixY, 80, 'r', 'x')
            plt.grid()

            plt.show()

        ################
        match_found = 1
        non_match_found = 1


doublette('Cherry')






