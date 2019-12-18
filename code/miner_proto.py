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
    cam_distance_threshold = 2.
    frame_size = 30

    match_found = 0
    non_match_found = 0
    first_img_found = 0
    ############################

    fx, fy, cx, cy = get_cam_intrinsics()
    drone_cam_translation = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.46], [0., 0., 0., 1.]])
    print('fx, fy, cx, cy...............', fx, fy, cx, cy)

    rotmat_x = build_rot_mat(-.5*np.pi, 'x')
    rotmat_y = build_rot_mat(-.5*np.pi, 'y')


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

            '''find random initial frame'''
            snapshot1_index = np.random.randint(num_frames)

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

        rotmat_z1 = build_rot_mat(angle1, 'z')

        lens1_position = np.dot(rotmat_z1, np.dot(rotmat_x, np.dot(rotmat_y, drone_cam_translation[:3,3])) * np.array([-1.,1., 1.])) + np.transpose(cam_position1[:3])
        x1, y1, z1 = lens1_position[:3]

        
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

            angle2 = -2 * np.arccos(cam_position2[-1])

            rotmat_z2 = build_rot_mat(angle2, 'z')

            lens2_position = np.dot(rotmat_z2, np.dot(rotmat_x, np.dot(rotmat_y, drone_cam_translation[:3,3])) * np.array([-1.,1., 1.])) + np.transpose(cam_position2[:3])
            x2, y2, z2 = lens2_position[:3]

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
        translation = np.transpose(lens2_position - lens1_position)

        '''compute sigma1 -> point coordinates'''
        cam1_2_point_distance = depthImg1[rand_row, rand_col]
        d1 = cam1_2_point_distance

        '''translate row, col to cam-coords'''
        u1, v1 = rand_col, 2 * cy - rand_row

        alpha1 = (u1 - cx) * d1 / fx
        beta1 = (v1 - cy) * d1 / fy
        gamma1 = d1

        sigma1 = np.transpose(np.array([alpha1, beta1, gamma1, 1.]))


        sigma1_in_worldcoords = np.dot(rotmat_x, np.dot(rotmat_y, sigma1[:3])) * np.array([-1., 1., 1.])
        # sigma1_in_worldcoords = np.dot(rotmat_x, np.dot(rotmat_y, sigma1[:3])) * np.array([1., 1., -1.]) # delte y rotation

        real_point = np.dot(rotmat_z1, sigma1_in_worldcoords) + np.transpose(lens1_position)

        lens2_2realpoint = real_point - lens2_position

        '''bis hier stimmts!!!!!!!!'''




########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


        '''should be correct up to this point!!!'''
        ''' sigma_1 = point in camera1 coordinate system: x: horizontal, y: vertical, z: 'depth' '''
        ''' translation = lens2_position - lens1_position '''

        '''rotate pc1 (sigma1) around x by pi/2 '''
        sigma1_x_rotated = np.dot(build_rot_mat(.5*np.pi, 'x'), sigma1[:3])

        '''flip z-axis'''
        sigma1_x_rotated_flipped_z = np.transpose(np.array([sigma1_x_rotated[0], sigma1_x_rotated[1], -1*sigma1_x_rotated[2]]))

        inv_rotation_angle1 = np.linalg.inv(build_rot_mat(angle1, 'z'))

        Pw_notranslation = np.dot(inv_rotation_angle1, sigma1_x_rotated_flipped_z)

        Pw = Pw_notranslation - translation

        rotation_angle2 = build_rot_mat(angle2, 'z')
        PwCam2 = np.dot(rotation_angle2, Pw)

        PwCam2_flipped = np.array([PwCam2[0], PwCam2[1], -1*PwCam2[2]])
        PwCam2_flipped_back_rotated = np.dot(np.linalg.inv(build_rot_mat(.5*np.pi, 'x')), PwCam2_flipped)

        print()
        print('sigma1_x_rotated')
        print(sigma1_x_rotated)
        print()
        print('sigma1_x_rotated_flipped_z')
        print(sigma1_x_rotated_flipped_z)
        print()
        print('inv_rotation_angle1')
        print(inv_rotation_angle1)
        print()
        print('Pw_notranslation')
        print(Pw_notranslation)
        print()
        print('Pw')
        print(Pw)
        print()
        print('PwCam2')
        print(PwCam2)
        print()
        print('PwCam2_flipped_back_rotated')
        print(PwCam2_flipped_back_rotated)




        rotated_translation = np.dot(build_rot_mat(1*(angle2), 'z'), translation)
        print('rotated_translation')
        print(rotated_translation)
        print('quaternion_cam2.rotate(translation)')
        print(quaternion_cam2.rotate(translation))


        sigma2 = np.dot(build_rot_mat(1*(angle2 - angle1), 'y'), sigma1[:3])# - np.dot(build_rot_mat(angle2, 'z'), translation)

        sigma2 = quaternion_cam1.inverse.rotate(sigma1_in_worldcoords) 


        # alpha2 = sigma2[0] + rotated_translation[1]
        # beta2 = sigma2[1] + rotated_translation[2] #always correct
        # gamma2 = sigma2[2] - rotated_translation[0]

        
        alpha2 = PwCam2_flipped_back_rotated[0]
        beta2 = PwCam2_flipped_back_rotated[1]
        gamma2 = PwCam2_flipped_back_rotated[2]

        d2 = gamma2



        u2 = (alpha2 * fx / d2) + cx
        v2 = (beta2 * fy / d2) + cy

        col2 = u2
        row2 = 2 * cy - v2



        sigma22 = np.transpose(np.array([alpha2, beta2, gamma2, 1.]))

        sigma2_in_worldcoords = np.dot(rotmat_x, np.dot(rotmat_y, sigma22[:3])) * np.array([-1., 1., 1.])

        real_point2 = np.dot(rotmat_z2, sigma2_in_worldcoords) + np.transpose(lens2_position)



        ptCam = sigma22[:3]



        # further processing

        # Compute bounding box in pixel coordinates
        bboxRange = np.array([
            [ptCam[0]-voxelGridPatchRadius*voxelSize, ptCam[0]+voxelGridPatchRadius*voxelSize], 
            [ptCam[1]-voxelGridPatchRadius*voxelSize, ptCam[1]+voxelGridPatchRadius*voxelSize],
            [ptCam[2]-voxelGridPatchRadius*voxelSize, ptCam[2]+voxelGridPatchRadius*voxelSize]])
        
        bboxCorners = np.array([
            [bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1]],
            [bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1],bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1]],
            [bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1]]])

        bboxRange = np.reshape(bboxRange,(3,2))

        bboxCorners = np.reshape(bboxCorners,(3,8))

        p1_bboxCornersCam = bboxCorners



        if output:
            print('translation..............', np.round(translation, 2))
            print('angle1...................', np.round(angle1, 2))
            print('angle2...................', np.round(angle2, 2))
            print('angle delta..............', np.round(angle_delta, 2))
            print()
            print('u1, v1...................', u1, v1)
            print('sigma1...................', np.round(sigma1, 2))
            print('sigma1_in_worldcoords....', np.round(sigma1_in_worldcoords, 2))
            print('real_point...............', np.round(real_point, 2))
            print()
            print('real_point2..............', np.round(real_point2, 2))
            print()
            print('point dist...............', np.round(dist_calc(real_point, real_point2), 2))
            print()
            print('alpha2, beta2, gamma2....', np.round(np.array([alpha2, beta2, gamma2]), 2))
            print('sigma2...................', np.round(sigma2, 2))
            print('sigma2_in_worldcoords....', np.round(sigma2_in_worldcoords, 2))
            print('lens2_2realpoint.........', np.round(lens2_2realpoint, 2))
            print('u2, v2...................', np.round(u2), np.round(v2))
            print('row2, col2...............', np.round(row2), np.round(col2))
            print()
            print('=======================================================> transformation done')
            print()

        if plots:

            '''load respective depth img'''
            depthImg2 = convert_pfm(random_tree_path + file_index2 + 'pl.pfm')
            depthImg2[depthImg2 > depth_limit] = 0


            real_point_fromcam2 = np.dot(build_rot_mat(angle2, 'z'), sigma2_in_worldcoords) + np.transpose(lens2_position)

            test_mat1 = build_rot_mat(angle1, 'z')
            test_mat2 = build_rot_mat(angle2, 'z')
            # test_mat1 = build_trafo_mat(x1, y1, z1, angle1)
            # test_mat2 = build_trafo_mat(x2, y2, z2, angle2)

            optical_axis1 = np.dot(test_mat1[0:3,0:3],np.transpose(np.array([gamma1, 0., 0.]))) + np.transpose(lens1_position)
            optical_axis2 = np.dot(test_mat2[0:3,0:3],np.transpose(np.array([gamma2, 0., 0.]))) + np.transpose(lens2_position)

            optical_axis1_x = np.linspace(optical_axis1[0], lens1_position[0], 10)
            optical_axis1_y = np.linspace(optical_axis1[1], lens1_position[1], 10)
            optical_axis1_z = np.linspace(optical_axis1[2], lens1_position[2], 10)
            
            optical_axis2_x = np.linspace(optical_axis2[0], lens2_position[0], 10)
            optical_axis2_y = np.linspace(optical_axis2[1], lens2_position[1], 10)
            optical_axis2_z = np.linspace(optical_axis2[2], lens2_position[2], 10)
            
            point2cam1_x = np.linspace(real_point[0], lens1_position[0], 10)
            point2cam1_y = np.linspace(real_point[1], lens1_position[1], 10)
            point2cam1_z = np.linspace(real_point[2], lens1_position[2], 10)
   
            point2cam2_x = np.linspace(real_point[0], lens2_position[0], 10)
            point2cam2_y = np.linspace(real_point[1], lens2_position[1], 10)
            point2cam2_z = np.linspace(real_point[2], lens2_position[2], 10)
   




            plt.figure()
            plt.plot(data_table[:,1], data_table[:,2], c='g', label='drone flight')
            plt.scatter(x1, y1, c='r', label='cam_position1')
            plt.scatter(x2, y2, c='b', label='cam_position2')
            plt.scatter(x2, y2, c='b', label='cam_position2')
            plt.plot(optical_axis1_x, optical_axis1_y, c='k', label='center1')
            plt.plot(point2cam1_x, point2cam1_y, c='r', label='sight1')
            plt.plot(point2cam2_x, point2cam2_y, c='b', label='sight2')
            plt.plot(optical_axis2_x, optical_axis2_y, c='k', label='center2')            
            plt.scatter(real_point[0], real_point[1], c='r', label='real_point1')
            plt.grid()
            plt.legend()


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x1, y1, z1, c='r', label='cam_position1')
            ax.scatter(x2, y2, z2, c='b', label='cam_position2')

            ax.scatter(bboxCorners[0,:] + real_point[0], bboxCorners[1,:] + real_point[1], bboxCorners[2,:] + real_point[2], c='g', label='corners')

            ax.plot(data_table[:,1], data_table[:,2], data_table[:,3], c='y', label='drone flight')

            # ax.plot(trans_vec_x, trans_vec_y, trans_vec_z, c='m', label='translation')

            ax.plot(optical_axis1_x, optical_axis1_y, optical_axis1_z, c='k', label='center1')
            ax.plot(point2cam1_x, point2cam1_y, point2cam1_z, c='r', label='sight1')
            ax.plot(point2cam2_x, point2cam2_y, point2cam2_z, c='b', label='sight2')

            ax.plot(optical_axis2_x, optical_axis2_y, optical_axis2_z, c='k', label='center2')

            ax.scatter(real_point[0], real_point[1], real_point[2], c='g', label='real_point1')
            ax.scatter(real_point2[0], real_point2[1], real_point2[2], c='b', label='real_point2')
            # ax.scatter(real_point_fromcam2[0], real_point_fromcam2[1], real_point_fromcam2[2], c='g', label='real_point2')

            
            for p in range(-60, 0):
                if p == -10:
                    ax.scatter(mean_x, mean_y, p/10., c='k', label='tree')
                else:
                    ax.scatter(mean_x, mean_y, p/10., c='k')

            ax.legend()

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            # ax.set_xlim(-10, 10.)
            # ax.set_ylim(-10, 10.)
            # ax.set_zlim(-10., 0.)


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
            # plt.scatter(u2, v2, 80, 'y', 'x')
            # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
            # plt.scatter(pixX, pixY, 80, 'r', 'x')
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
            # plt.scatter(u2, v2, 80, 'y', 'x')
            # plt.scatter(pixX, 2 * cy - pixY, 80, 'r', 'x')
            # plt.scatter(pixX, pixY, 80, 'r', 'x')
            plt.grid()

            plt.show()

        ################
        match_found = 1
        non_match_found = 1


doublette('Cherry')






