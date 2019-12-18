import os
import glob
from numpy import array
#from PIL import Image
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import spatial
from scipy.spatial import cKDTree
import math
import time
from matplotlib import pyplot as plt
import re
import sys
from struct import *
import imageio
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D

first_img = 14400
last_img = 15119


def read_txt(file):
    
    with open(file, 'r') as f:

        f = f.readlines()
        output = []

        for line in f:  

            line = line.split(' ')
            output.append(line)
            
        return output
img_data = np.asarray(read_txt('TreeData/image data.txt'))

def dist_calc(coords, other_coords):
    
    xdistance = np.power(coords[0] - other_coords[0], 2)
    ydistance = np.power(coords[1] - other_coords[1], 2)
    zdistance = np.power(coords[2] - other_coords[2], 2)
    
    dist = np.sqrt(xdistance + ydistance + zdistance)
    
    return dist

def get_cam_intrinsics(which):
    intrinsics = read_txt('camera-intrinsics.txt')
    intrinsics = np.asarray(intrinsics)
    intrinsics = intrinsics[:,[1, 3, 5]]

    lukas_intrinsics = np.array([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
    yifei_intrinsics = np.array([[463., 0., 320.], [0., 463., 240.], [0., 0., 1.]])

    print()
    print('Cam intrinsics yifei:')
    print(yifei_intrinsics)
    print()
    print('Cam intrinsics lukas:')
    print(lukas_intrinsics)


    for i in range(3):
        for j in range(3):
            intrinsics[i,j] = str(intrinsics[i,j])[:-2]

    if which == 'yifei':
        return yifei_intrinsics
    elif which == 'lukas':
        return lukas_intrinsics

def get_rotation_matrix(angle, axis):
    
    if axis == 'x': #up and down rotation
        
        mat = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]])
    
    if axis == 'y': #left and right rotation
        
        mat = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]])
    
    return mat

def convert_pfm(file, plot=True, debug=False):

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

        image = np.reshape(np.asarray(log_img), (480, -1))
        
        if plot:
            plt.figure(figsize=(8,16))
            plt.imshow(image)
            plt.title('log depth img')
        
        return np.asarray(img)

def get_xy_from_uv(u, v, rand_row, rand_col, depthImg):
    
    dist = depthImg[rand_row,rand_col]
    
    cam_intrinsics = get_cam_intrinsics('yifei')
    f_x = cam_intrinsics[0,0]
    f_y = cam_intrinsics[1,1]
    u_0 = cam_intrinsics[0,2]
    v_0 = cam_intrinsics[1,2]
    
    x = (u - u_0) / f_x * dist
    y = (v - v_0) / f_y * dist
    z = dist
    
    return np.array([x, y, z])

def get_uv_from_xy(transformation12, point1_from_cam1):
    
    cam_intrinsics = get_cam_intrinsics()
    f_x = cam_intrinsics[0,0]
    f_y = cam_intrinsics[1,1]
    u_0 = cam_intrinsics[0,2]
    v_0 = cam_intrinsics[1,2]
    
    vec1 = np.zeros((4,1))
    vec1[:3,0] = point1_from_cam1
    vec1[3,0] = 1

    point2_from_cam1 = np.matmul(transformation12, vec1)
    
    x, y= point2_from_cam1[:2]
    
    d = point2_from_cam1[2]
    u = u_0 + x * f_x / d
    v = v_0 + y * f_y / d
    
    
    return u, v, np.float32(d), point2_from_cam1

def get_cam_position(img_number):
    
    data_table = np.asarray(pd.read_table('TreeData/poses.txt', sep='\s', header=0, index_col=False, engine='python'))
    # print(pd.read_table('TreeData/poses.txt', sep='\s', header=0, index_col=False, engine='python'))
    index = np.squeeze(np.where(data_table[:,0] == img_number))
    #index = index[0]
    index_data = data_table[index]

    return index_data[1:]

def get_cam_position2(img_data, img_number):
    
    specs = np.asarray(img_data)[img_number + 2]
    
    x_coord = specs[1]
    y_coord = specs[2]
    z_coord = specs[3]
    
    coords = np.transpose(np.array([x_coord, y_coord, z_coord]))
    
    return np.asarray(coords, np.float32)
        
def transfrom_quaternions(img_number):
    
    quats = get_cam_position(img_number)[3:]

    q_x = quats[0]
    q_y = quats[1]
    q_z = quats[2]
    q_w = quats[3]

    # q_0 = quats[3]
    # q_1 = quats[0]
    # q_2 = quats[1]
    # q_3 = quats[2]
    
    #print(quats)
    
    R = np.zeros((3,3))
    
    R[0,0] = 1 - 2 * np.power(q_y, 2) - 2 * np.power(q_z, 2)
    R[1,1] = 1 - 2 * np.power(q_x, 2) - 2 * np.power(q_z, 2)
    R[2,2] = 1 - 2 * np.power(q_x, 2) - 2 * np.power(q_y, 2)
    
    R[0,1] = 2 * (q_x* q_y - q_z * q_w)
    R[0,2] = 2 * (q_x * q_z + q_y * q_w)
    R[1,0] = 2 * (q_x * q_y + q_z * q_w)
    R[1,2] = 2 * (q_y * q_z - q_x * q_w)
    R[2,0] = 2 * (q_x * q_z - q_y * q_w)
    R[2,1] = 2 * (q_y * q_z + q_x * q_w)
    
    # R[0,0] = np.power(q_0, 2) + np.power(q_1, 2) - np.power(q_2, 2) - np.power(q_3, 2)
    # R[1,1] = np.power(q_0, 2) - np.power(q_1, 2) + np.power(q_2, 2) - np.power(q_3, 2)
    # R[2,2] = np.power(q_0, 2) - np.power(q_1, 2) - np.power(q_2, 2) + np.power(q_3, 2)
    
    # R[0,1] = 2 * (q_1 * q_2 - q_0 * q_3)
    # R[0,2] = 2 * (q_0 * q_2 + q_1 * q_3)
    # R[1,0] = 2 * (q_1 * q_2 + q_0 * q_3)
    # R[1,2] = 2 * (q_2 * q_3 - q_0 * q_1)
    # R[2,0] = 2 * (q_1 * q_3 - q_0 * q_2)
    # R[2,1] = 2 * (q_0 * q_1 + q_2 * q_3)
    
    return R

def trafo(which_file, which_intrinsics, snapshot1_index, snapshot2_index, u1, v1, angle, d1):

                if which_file == 'poses':

                    x1, y1, z1 = get_cam_position(snapshot1_index)[:3] #from poses.txt
                    x2, y2, z2 = get_cam_position(snapshot2_index)[:3]

                if which_file == 'data':

                    x1, y1, z1 = get_cam_position2(img_data, snapshot1_index)[0], get_cam_position2(img_data, snapshot1_index)[1], get_cam_position2(img_data, snapshot1_index)[2]
                    x2, y2, z2 = get_cam_position2(img_data, snapshot2_index)[0], get_cam_position2(img_data, snapshot2_index)[1], get_cam_position2(img_data, snapshot2_index)[2]

                if which_intrinsics == 'lukas':

                    cam_intrinsics = get_cam_intrinsics('lukas')
                    f_x = cam_intrinsics[0,0]
                    f_y = cam_intrinsics[1,1]
                    u_0 = cam_intrinsics[0,2]
                    v_0 = cam_intrinsics[1,2]

                if which_intrinsics == 'yifei':

                    cam_intrinsics = get_cam_intrinsics('yifei')
                    f_x = cam_intrinsics[0,0]
                    f_y = cam_intrinsics[1,1]
                    u_0 = cam_intrinsics[0,2]
                    v_0 = cam_intrinsics[1,2]


                # print()
                # print('============================================================================')
                # print()

                # print('x1, y1, z1-------------->', np.round(x1, 3), np.round(y1, 3), np.round(z1, 3))
                # print()
                # print('x2, y2, z2-------------->', np.round(x2, 3), np.round(y2, 3), np.round(z2, 3))
                # print()

                delta_x = x2 - x1
                delta_y = y2 - y1
                delta_z = z2 - z1

                # angle = angle2 - angle1

                # u1, v1 = rand_col, 2 * v_0 - rand_row
                # u1, v1 = rand_col, rand_row
                # d1 = dist_p1



                trafo_mat = np.zeros((4,4))

                trafo_mat[0,0] = np.cos(angle)
                trafo_mat[0,1] = -np.sin(angle)
                trafo_mat[0,2] = 0.
                trafo_mat[0,3] = delta_x

                trafo_mat[1,0] = np.sin(angle)
                trafo_mat[1,1] = np.cos(angle)
                trafo_mat[1,2] = 0.
                trafo_mat[1,3] = delta_y

                trafo_mat[2,0] = 0.
                trafo_mat[2,1] = 0.
                trafo_mat[2,2] = 1.
                trafo_mat[2,3] = delta_z

                trafo_mat[3,0] = 0.
                trafo_mat[3,1] = 0.
                trafo_mat[3,2] = 0.
                trafo_mat[3,3] = 1.

                alpha1 = (u1 - u_0) * d1 / f_x
                beta1 = (v1 - v_0) * d1 / f_y
                gamma1 = d1

                sigma1 = np.transpose(np.array([alpha1, beta1, gamma1, 1.]))

                sigma2 = np.matmul(trafo_mat, sigma1)

                alpha2 = sigma2[0]
                beta2 = sigma2[1]
                gamma2 = sigma2[2]

                d2 = gamma2

                u2 = (alpha2 * f_x / d2) + u_0
                v2 = (beta2 * f_y / d2) + v_0

                col2 = u2
                row2 = 2 * v_0 - v2

                return col2, row2
    
def doublette(plots=True, output=True):
    
    depth_limit = 8.
    cam_distance_threshold = 3.
    
    #poses = read_txt('TreeData/poses.txt')
    img_data = np.asarray(read_txt('TreeData/image data.txt'))
    
    data = np.zeros((img_data.shape[0]-2, 6))
    tree_number = np.zeros(img_data.shape[0]-2)
    tree_kind = []

    for i in range(img_data.shape[0]-4):

        data[i,:] = np.asarray(img_data[i+2])[:6]
        tree_kind.append(np.asarray(img_data[i+2])[6])
        tree_number[i] = np.asarray(img_data[i+2])[7]

    cam_intrinsics = get_cam_intrinsics('lukas')
    f_x = cam_intrinsics[0,0]
    f_y = cam_intrinsics[1,1]
    u_0 = cam_intrinsics[0,2]
    v_0 = cam_intrinsics[1,2]
    print('camera intrinsics:')
    print('f_x: ', f_x, ' , f_y: ', f_y)
    print('u_0: ', u_0, ', v_0: ', v_0)
    
    frame_size = 30
    
    match_found = 0
    non_match_found = 0

    
    while (match_found < 1 and non_match_found < 1):
        
        print()
        print('finding initial camera position with visible tree..........................')
        print()
        
        '''pick a random image number for snapshot1 and read pose'''
        snapshot1_index = np.random.randint(first_img, last_img)


        #######################
        #######################
        #######################
        # snapshot1_index = 14751
        ################
        #######
        #######################
        #######################
                

        cam_position1 = get_cam_position(snapshot1_index)
        alt_cam_position1 = get_cam_position2(img_data, snapshot1_index)

        axis1 = cam_position1[3:6]
        if np.round(cam_position1[3], 2) == 0. and np.round(cam_position1[4], 2) == 0. and np.round(cam_position1[5], 2) == 0.:
            axis1 = np.array([1e-04, 1e-04, 1e-04])

        quaternion_cam1 = Quaternion(axis=axis1, angle=cam_position1[6])
        quat_to_rot1 = quaternion_cam1.rotation_matrix
        angle1 = quaternion_cam1.radians

        '''load the depth file'''
        filename_depth1 = str('TreeData') + '/' + str(snapshot1_index) + 'pl.pfm'
        
        depthImg1 = convert_pfm(filename_depth1, plot=False, debug=False)
        
        depthImg1 = np.reshape(np.asarray(depthImg1), (480, -1))
        
        '''load the segmentation image'''
        filename_seg1 = str('TreeData') + '/' + str(snapshot1_index) + 'segLabel.png'
        seg_img1 = imageio.imread(filename_seg1)

        '''zero unwanted parts of depth img'''
        depthImg1[seg_img1 < 2] = 0 #delete non_tree parts
        depthImg1[depthImg1 > depth_limit] = 0 #delete too far away points
        
        '''check if there is a tree in sight'''
        if np.array(np.nonzero(depthImg1)).shape[1] < 2:
            print('no tree here...')
            break

        '''choose a random point on the tree in the frame'''
        if np.array(np.nonzero(depthImg1)).shape[1] >= 2:
            
            rand_row, rand_col = 0, 0
            
            dummy1 = 0
            
            while (rand_row < 2*frame_size or rand_row > (depthImg1.shape[0] - 2*frame_size)) or (rand_col < 2*frame_size or rand_col > (depthImg1.shape[1] - 2*frame_size)):
                
                dummy1 += 1
                random_spot1 = np.random.randint(0, np.array(np.nonzero(depthImg1)).shape[1])
                rand_row, rand_col = np.array(np.nonzero(depthImg1))[:,random_spot1]

            #######################
            #######################
            #######################
            # rand_row, rand_col = 256, 195
            #######################
            #######################
            #######################
                
            
            rand_part = seg_img1[rand_row, rand_col]
            dist_p1 = depthImg1[rand_row, rand_col]

            d1 = dist_p1

            '''get point position in camera reference system'''
            u, v = rand_col, 2 * v_0 - rand_row
            u1, v1 = rand_col, 2 * v_0 - rand_row
            
            point1_from_cam1 = get_xy_from_uv(u, v, rand_row, rand_col, depthImg1)

            if output:
                print('finding initial camera position with visible tree..........................')
                print('iterations needed to find tree.............................................', int(dummy1))
                print('frame index for image 1....................................................', snapshot1_index)
                print('random point in frame (u,v)................................................', np.array([rand_row, rand_col]))
                print('distance from camera.......................................................', np.round(dist_p1, 3))
                print()
                print('cam1 position and quaternion:')
                print('x, y, z....................................................................', np.round(cam_position1[:3], 3))
                print('quaternion.................................................................', np.round(cam_position1[3:], 3))
                # print()
                # print('point1 position in camera1 reference system:')
                # print('x, y, z....................................................................', np.round(point1_from_cam1, 2))
                print()
                print('==========================================> 1st checkpoint: relevant initial point found')
        
        point_distance = 100
        loop_dummy = 0
        dummy2 = 0
        dummy3 = 1
        
        print()
        print('finding 2nd camera in relevant distance....................................')

        while point_distance > cam_distance_threshold:
            
            loop_dummy += 1

            '''2nd random frame'''
            snapshot2_index = snapshot1_index
            while snapshot2_index == snapshot1_index:
                snapshot2_index = np.random.randint(first_img, last_img)


            #######################
            #######################
            #######################
            # snapshot2_index = 14663
            #######################
            #######################
            #######################
                
            # cam_position2 = data[snapshot2_index, 1:]
            cam_position2 = get_cam_position(snapshot2_index)
            alt_cam_position2 = get_cam_position2(img_data, snapshot2_index)


            if len(cam_position2) < 3:
                continue

            axis2 = cam_position2[3:6]
            if cam_position2[3] == 0. and cam_position2[4] == 0. and cam_position2[5] == 0.:
                axis2 = np.array([1e-04, 1e-04, 1e-04])

            quaternion_cam2 = Quaternion(axis=axis2, angle=cam_position2[6])
            quat_to_rot2 = quaternion_cam2.rotation_matrix
        
            angle2 = quaternion_cam2.radians

            angle = angle2 - angle1


            # img | x | y | z | qx | qy | qz | qw
            
            '''check if camera position is close enough'''
            # point_distance = dist_calc(cam_position1[:3], cam_position2[:3])
            point_distance = dist_calc(alt_cam_position1, alt_cam_position2)

            if point_distance > cam_distance_threshold:
                continue
            
            #print('point_distance..................', np.round(point_distance, 2))

            '''load the depth file'''
            filename_depth2 = str('TreeData') + '/' + str(snapshot2_index) + 'pl.pfm'
            depthImg2 = convert_pfm(filename_depth2, plot=False, debug=False)
            #depthImg1[depthImg1 > 8] = 0

            depthImg2 = np.reshape(np.asarray(depthImg2), (480, -1))
            
            '''load the segmentation image'''
            filename_seg2 = str('TreeData') + '/' + str(snapshot2_index) + 'segLabel.png'
            seg_img2 = imageio.imread(filename_seg2)

            '''zero unwanted parts of depth img'''
            depthImg2[seg_img2 < 2] = 0 #delete non_tree parts
            depthImg2[depthImg2 > depth_limit] = 0 #delete too far away points
            
            '''check if there is a tree in sight'''
            if np.array(np.nonzero(depthImg2)).shape[1] < 2:
                dummy2 += 1
                continue

            '''compute translation and rotation '''
            full_rotation_cam1 = transfrom_quaternions(snapshot1_index)
            full_rotation_cam2 = transfrom_quaternions(snapshot2_index)

            full_rotation = np.dot(full_rotation_cam2, np.linalg.inv(full_rotation_cam1))

            # translation = cam_position2[:3]-cam_position1[:3]
            translation = alt_cam_position2 - alt_cam_position1
            
            transformation12 = np.zeros((4,4))
            transformation12[:3,:3] = quat_to_rot2
            transformation12[-1,-1] = 1
            transformation12[:3,-1] = translation
            
            # u2, v2, d2, point2_from_cam1 = get_uv_from_xy(transformation12, point1_from_cam1)

            # row2 = depthImg2.shape[0] - v2
            # col2 = u2


            '''trafo'''

            # def trafo(which_file, which_intrinsics, snapshot1_index, snapshot2_index, u1, v1, angle, d1):

            #     if which_file == 'poses':

            #         x1, y1, z1 = get_cam_position(snapshot1_index)[:3] #from poses.txt
            #         x2, y2, z2 = get_cam_position(snapshot2_index)[:3]

            #     if which_file == 'data':

            #         x1, y1, z1 = get_cam_position2(img_data, snapshot1_index)[0], get_cam_position2(img_data, snapshot1_index)[1], get_cam_position2(img_data, snapshot1_index)[2]
            #         x2, y2, z2 = get_cam_position2(img_data, snapshot2_index)[0], get_cam_position2(img_data, snapshot2_index)[1], get_cam_position2(img_data, snapshot2_index)[2]

            #     if which_intrinsics == 'lukas':

            #         cam_intrinsics = get_cam_intrinsics('lukas')
            #         f_x = cam_intrinsics[0,0]
            #         f_y = cam_intrinsics[1,1]
            #         u_0 = cam_intrinsics[0,2]
            #         v_0 = cam_intrinsic

            #     if which_intrinsics == 'yifei':

            #         cam_intrinsics = get_cam_intrinsics('yifei')
            #         f_x = cam_intrinsics[0,0]
            #         f_y = cam_intrinsics[1,1]
            #         u_0 = cam_intrinsics[0,2]
            #         v_0 = cam_intrinsics[1,2]


            #     # print()
            #     # print('============================================================================')
            #     # print()

            #     # print('x1, y1, z1-------------->', np.round(x1, 3), np.round(y1, 3), np.round(z1, 3))
            #     # print()
            #     # print('x2, y2, z2-------------->', np.round(x2, 3), np.round(y2, 3), np.round(z2, 3))
            #     # print()

            #     delta_x = x2 - x1
            #     delta_y = y2 - y1
            #     delta_z = z2 - z1

            #     # angle = angle2 - angle1

            #     # u1, v1 = rand_col, 2 * v_0 - rand_row
            #     # u1, v1 = rand_col, rand_row
            #     # d1 = dist_p1



            #     trafo_mat = np.zeros((4,4))

            #     trafo_mat[0,0] = np.cos(angle)
            #     trafo_mat[0,1] = -np.sin(angle)
            #     trafo_mat[0,2] = 0.
            #     trafo_mat[0,3] = delta_x

            #     trafo_mat[1,0] = np.sin(angle)
            #     trafo_mat[1,1] = np.cos(angle)
            #     trafo_mat[1,2] = 0.
            #     trafo_mat[1,3] = delta_y

            #     trafo_mat[2,0] = 0.
            #     trafo_mat[2,1] = 0.
            #     trafo_mat[2,2] = 1.
            #     trafo_mat[2,3] = z2 - z1

            #     trafo_mat[3,0] = 0.
            #     trafo_mat[3,1] = 0.
            #     trafo_mat[3,2] = 0.
            #     trafo_mat[3,3] = 1.

            #     alpha1 = (u1 - u_0) * d1 / f_x
            #     beta1 = (v1 - v_0) * d1 / f_y
            #     gamma1 = d1

            #     sigma1 = np.transpose(np.array([alpha1, beta1, gamma1, 1.]))

            #     sigma2 = np.dot(trafo_mat, sigma1)

            #     alpha2 = sigma2[0]
            #     beta2 = sigma2[1]
            #     gamma2 = sigma2[2]

            #     d2 = gamma2

            #     u2 = (alpha2 * f_x / d2) + u_0
            #     v2 = (beta2 * f_y / d2) + v_0

            #     col2 = u2
            #     row2 = 2 * v_0 - v2

            #     return col2, row2

            # if (col2 < 1*frame_size or col2 > (depthImg1.shape[1] - 1*frame_size)) or (row2 < 1*frame_size or row2 > (depthImg1.shape[0] - 1*frame_size)):
            #     if dummy3 == 1:
            #         print('corresponding point not in camera frustum...')
            #     point_distance = 100.
            #     dummy3 += 1
            #     continue

            newcol1, newrow1 = trafo('poses', 'lukas', snapshot1_index, snapshot2_index, u1, v1, angle, d1)
            newcol2, newrow2 = trafo('data', 'lukas', snapshot1_index, snapshot2_index, u1, v1, angle, d1)
            newcol3, newrow3 = trafo('poses', 'yifei', snapshot1_index, snapshot2_index, u1, v1, angle, d1)
            newcol4, newrow4 = trafo('data', 'yifei', snapshot1_index, snapshot2_index, u1, v1, angle, d1)

            # print('u1, v1, d1-------------->', np.round(u1, 3), np.round(v1, 3), np.round(d1, 3))
            # print()
            # print('v0, u0, f_x, f_y', v_0, u_0, f_x, f_y)
            # print()
            # print('angle------------------->', np.round(angle, 3))
            # print()
            # print('trafo:')
            # print(np.round(trafo_mat, 3))
            # print()
            # print('sigma1------------------>', np.round(sigma1, 3))
            # print()
            # print('sigma2------------------>', np.round(sigma2, 3))
            # print()
            # print('d2---------------------->', np.round(d2, 3))
            # print()
            # print('u2, v2------------------>', np.round(u2, 3), np.round(v2, 3))
            # print()
            # print('col2, row2-------------->', np.round(col2, 3), np.round(row2, 3))
            # print()


            print()
            print('============================================================================')
            print()
            print('============================================================================')
            print()
            print('============================================================================')
            print()







        if output:

            print('iterations needed to find image with tree..................................' + str(dummy2))
            print()
            print('iterations needed to find point in frustum.................................' + str(dummy3))
            print()
            print(str(loop_dummy) + ' iterations needed | distance to initial position........', np.round(point_distance, 3))
            print('frame index for image 2....................................................', snapshot2_index)
            print()
            print('2nd cam position and quaterion:')
            print('x, y, z....................................................................', np.round(cam_position2[:3], 3))
            print('quaternion.................................................................', np.round(cam_position2[3:], 3))
            # print()
            # print('point2 position in camera1 reference system:')
            # print('x, y, z....................................................................', np.round(np.reshape(point2_from_cam1[:3], -1), 2))
            # print()
            # print('u2, v2, d2.................................................................', u2, v2, d2)
            # print('full transformation matrix:')
            # print(np.round(transformation12, 2))
            # print()
            # print('quaternion trafo 1:')
            # print(quat_to_rot1)
            # print()
            # print('quaternion trafo 2:')
            # print(quat_to_rot2)
            # print()
            # print('my trafo 1:')
            # print(transfrom_quaternions(snapshot1_index))
            # print()
            print('angle1:', np.round(angle1, 3), '[radians] /', np.round(angle1/360*2*np.pi, 3), '[degree]')
            print('angle2:', np.round(angle2, 3), '[radians] /', np.round(angle2/360*2*np.pi, 3), '[degree]')
            print()
            print('==========================================> 2nd checkpoint: other camera position in proximity found')

        if plots:

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # For each set of style and range settings, plot n random points in the box
            # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
            # # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
            # #     xs = randrange(n, 23, 32)
            # #     ys = randrange(n, 0, 100)
            # #     zs = randrange(n, zlow, zhigh)
            # ax.scatter(x1, y1, z1, c='r', label='cam_position1')
            # ax.scatter(x2, y2, z2, c='b', label='cam_position2')
            
            # for p in range(-80, 0):
            #     if p == 50:
            #         ax.scatter(0., 0., p/10., c='k', label='tree')
            #     else:
            #         ax.scatter(0., 0., p/10., c='k')

            # ax.legend()


            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')

            # ax.set_xlim(-10, 10.)
            # ax.set_ylim(-10, 10.)
            # ax.set_zlim(-10., 0.)

            # plt.show()


            pic_title = []

            if int(rand_part) == 2:
                pic_title = 'trunk'
            elif int(rand_part) == 3:
                pic_title = 'branch'
            elif int(rand_part) == 4:
                pic_title = 'twig'
            elif int(rand_part) == 5:
                pic_title = 'leaf'
            else:
                pic_title = 'unsure... label: ' + str(rand_part)

            testimage1 = imageio.imread('TreeData/' + str(snapshot1_index) + '.png')
            testimage2 = imageio.imread('TreeData/' + str(snapshot2_index) + '.png')

            plt.figure(figsize=(10,10))
            
            plt.subplot(221)
            plt.imshow(depthImg1)
            # plt.scatter(rand_col, rand_row, c="r")
            plt.scatter(u1, 2*v_0 - v1, c="r")
            plt.scatter(rand_col, rand_row, c="r")
            #plt.scatter(np.array(np.nonzero(depthImg1))[1,:],np.array(np.nonzero(depthImg1))[0,:])
            plt.title('considered point:' + str(pic_title))
            
            plt.subplot(222)
            plt.imshow(depthImg2)
            plt.title('corresponding view point')
            # plt.scatter(col2, row2, c="r")
            plt.scatter(newcol1, newrow1, c="w", label='P/L')
            plt.scatter(newcol2, newrow2, c="r", label='D/L')
            plt.scatter(newcol3, newrow3, c="b", label='P/Y')
            plt.scatter(newcol4, newrow4, c="k", label='D/Y')

            # plt.scatter(u2, 2*v_0 - v2, c="w")

            plt.subplot(223)
            plt.imshow(testimage1)
            plt.title('image 1')
            plt.scatter(u1, 2*v_0 - v1, c="r")
            # plt.scatter(rand_col, rand_row, c="r")

            plt.subplot(224)
            plt.imshow(testimage2)
            plt.title('image 2')
            # plt.scatter(col2, row2, c="r")
            # plt.scatter(u2, 2*v_0 - v2, c="w")
            plt.scatter(newcol1, newrow1, c="w", label='P/L')
            plt.scatter(newcol2, newrow2, c="r", label='D/L')
            plt.scatter(newcol3, newrow3, c="b", label='P/Y')
            plt.scatter(newcol4, newrow4, c="k", label='D/Y')



            plt.show()
        

        ################
        match_found = 1
        non_match_found = 1



for i in range(10):
    doublette()






