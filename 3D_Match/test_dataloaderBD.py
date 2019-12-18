import os
import glob
from numpy import array
from PIL import Image
import numpy as np
from numpy.linalg import inv
from scipy import spatial
from scipy.spatial import cKDTree
import math

def getPair(sceneDataList, maxTries, voxelGridPatchRadius, voxelSize, voxelMargin):
    corresFound = 0
    nonMatchFound = 0
    while(corresFound < 1 and nonMatchFound < 1):
        # Pick a random scene and a set of random frames
        randSceneIdx = np.random.randint(len(sceneDataList))
        randFrameIdx = np.random.randint(len(sceneDataList[randSceneIdx]['frameList']), size=maxTries)
        camK = sceneDataList[randSceneIdx]['camK']

        # Find a random 3D point (in world coordinates) in a random frame
        framePrefix = sceneDataList[randSceneIdx]['frameList'][randFrameIdx[0]]
   
        ##################################################
        framePrefix = '/home/drzadmin/Desktop/3DMatch-pytorch/data/target/Cherry/seq-01/frame-000000'

        ###################################################
        p1_framePath = framePrefix
        depthIm = Image.open(framePrefix + '.depth.png')
        depthIm = np.array(depthIm)/1000       
        depthIm[depthIm > 6] = 0
        randDepthInd = np.random.randint(len(np.nonzero(depthIm)[0]))
        pixY,pixX = np.nonzero(depthIm)[0][randDepthInd], np.nonzero(depthIm)[1][randDepthInd]
        
        ####################################################
        pixX,pixY = 302, 248

        ####################################################
        p1_pixelCoords = np.array([pixX,pixY])

        ptCamZ = depthIm[pixY,pixX]
        pixY,pixX = pixY+1,pixX+1
        ptCamX = np.float((pixX-0.5-camK[0,2])*ptCamZ / camK[0,0])
        ptCamY = np.float((pixY-0.5-camK[1,2])*ptCamZ / camK[1,1])
        ptCam = np.array([[ptCamX],[ptCamY],[ptCamZ]])
        ####################################################        
        #print('camK',camK)
        #print('ptCam',ptCam)
        print('checkpoint1 pass')
        #######################################################

        p1_camCoords = ptCam

        with open(framePrefix + '.pose.txt', 'r') as f:
            f = f.readlines()
        extCam2World = []
        for line in f:           
            line = line.split()
            extCam2World.append(line)
       
        extCam2World = array(extCam2World).astype(float)
        
        p1CamLoc = extCam2World[0:3,3].T
        
        p1World = np.dot(extCam2World[0:3,0:3],ptCam) + np.reshape(extCam2World[0:3,3],(3,1))

        ###########################################################
        #print('p1World',p1World)
        print('checkpoint2 pass')
        ##############################################################
        
        # Compute bounding box in pixel coordinates
        bboxRange = np.array([[ptCam[0]-voxelGridPatchRadius*voxelSize, ptCam[0]+voxelGridPatchRadius*voxelSize], 
                [ptCam[1]-voxelGridPatchRadius*voxelSize, ptCam[1]+voxelGridPatchRadius*voxelSize],
                [ptCam[2]-voxelGridPatchRadius*voxelSize, ptCam[2]+voxelGridPatchRadius*voxelSize]])
        
        bboxCorners = np.array([[bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1]],
                [bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1],bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1]],
                [bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1]]])
        bboxRange = np.reshape(bboxRange,(3,2))
        bboxCorners = np.reshape(bboxCorners,(3,8))
        p1_bboxCornersCam = bboxCorners
        ##############################################################
        #print('bboxRange', bboxRange)
        #print('bboxCorners',bboxCorners)
        print('checkpoint3 pass')
        ##############################################################
        bboxPixX = np.round((bboxCorners[0,:]*camK[0,0]/bboxCorners[2,:])+camK[0,2])
        bboxPixY = np.round((bboxCorners[1,:]*camK[1,1]/bboxCorners[2,:])+camK[1,2])

        bboxPixX = np.array([pixX-max(np.abs(pixX-bboxPixX)), pixX+max(np.abs(pixX-bboxPixX))])
        bboxPixY = np.array([pixY-max(np.abs(pixY-bboxPixY)), pixY+max(np.abs(pixY-bboxPixY))])

        ############################################################3
        #print('bboxPixX',bboxPixX)
        #print('bboxPixY',bboxPixY)
        print('checkpoint4 pass')
        #############################################################
        
        if np.any(bboxPixX <= 0) or np.any(bboxPixX > 640) or np.any(bboxPixY <= 0) or np.any(bboxPixY > 480) :
            continue

        p1_bboxRangePixels = np.array([[bboxPixX],[bboxPixY]])
        p1_camK = camK
        p1 = {'bboxCornersCam': p1_bboxCornersCam, 'bboxRangePixels': p1_bboxRangePixels, 'framePath': p1_framePath,
                    'pixelCoords': p1_pixelCoords, 'camCoords': p1_camCoords, 'camK': p1_camK}

        # Loop through other random frames to find a positive point match
        for otherFrameIdx in range(1,len(randFrameIdx)):
            otherFramePrefix = sceneDataList[randSceneIdx]['frameList'][randFrameIdx[otherFrameIdx]]

            ###################################################
            otherFramePrefix = '/home/drzadmin/Desktop/3DMatch-pytorch/data/target/Cherry/seq-01/frame-000020'
            #####################################################

            #Check if 3D point is within camera view frustum
            with open(otherFramePrefix + '.pose.txt', 'r') as f:
                f = f.readlines()
            extCam2World = []
            for line in f:
                line = line.split()
                extCam2World.append(line)

            extCam2World = array(extCam2World).astype(float)
            '''
            if np.isnan(np.sum(extCam2World[:])):
                continue

            p2CamLoc = extCam2World[0:3,3]
            if np.sqrt(np.sum((p1CamLoc-p2CamLoc)**2)) < 1 :
                continue
			'''
            extWorld2Cam = inv(extCam2World)

            #########################################################
            #print('extWorld2Cam',extWorld2Cam)
            print('checkpoint5 pass')
            ##########################################################

            ptCam = np.dot(extWorld2Cam[0:3,0:3],p1World) + np.reshape(extWorld2Cam[0:3,3],(3,1))

            ########################################################
            #print('ptCam',ptCam)
            print('checkpoint6 pass')
            #########################################################


            pixX = np.round((ptCam[0]*camK[0,0]/ptCam[2])+camK[0,2]+0.5).astype(int)
            pixY = np.round((ptCam[1]*camK[1,1]/ptCam[2])+camK[1,2]+0.5).astype(int)

            #########################################################
            #print('pixX',pixX)
            #print('pixY',pixY)
            print('checkpoint7 pass')
            #########################################################
            
            if pixX > 0 and pixX <= 640 and pixY > 0 and pixY <= 480 :
                depthIm = Image.open(otherFramePrefix + '.depth.png')
                depthIm = np.array(depthIm)/1000       
                depthIm[depthIm > 6] = 0
                
                if abs(depthIm[pixY-1,pixX-1]-ptCam[2]) < 6 :     ########################!!!!! debug !!!!!! change to < 0.03 after################
                    ptCamZ = depthIm[pixY-1,pixX-1]
                    ptCamX = float((pixX-0.5-camK[0,2])*ptCamZ/camK[0,0])
                    ptCamY = float((pixY-0.5-camK[1,2])*ptCamZ/camK[1,1])
                    ptCam = np.array([[ptCamX],[ptCamY],[ptCamZ]])

                    ######################################################3
                    #print('ptCam',ptCam)
                    print('checkpoint8 pass')
                    #######################################################

                    # Compute bounding box in pixel coordinates
                    bboxRange = np.array([[ptCam[0]-voxelGridPatchRadius*voxelSize,ptCam[0]+voxelGridPatchRadius*voxelSize], 
                        [ptCam[1]-voxelGridPatchRadius*voxelSize,ptCam[1]+voxelGridPatchRadius*voxelSize],
                        [ptCam[2]-voxelGridPatchRadius*voxelSize,ptCam[2]+voxelGridPatchRadius*voxelSize]])
        
                    bboxCorners = np.array([[bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1]],
                        [bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1],bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1]],
                        [bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1]]])

                    bboxRange = np.reshape(bboxRange,(3,2))
                    bboxCorners = np.reshape(bboxCorners,(3,8))
        			##########################################################3
                    #print('bboxRange',bboxRange)
                    #print('bboxCorners',bboxCorners)
                    print('checkpoint9 pass')
                    #############################################################

                    p2_bboxCornersCam = bboxCorners
                    bboxPixX = np.round((bboxCorners[0,:]*camK[0,0]/bboxCorners[2,:])+camK[0,2])
                    bboxPixY = np.round((bboxCorners[1,:]*camK[1,1]/bboxCorners[2,:])+camK[1,2])

                    bboxPixX = np.array([pixX-max(np.abs(pixX-bboxPixX)), pixX+max(np.abs(pixX-bboxPixX))])
                    bboxPixY = np.array([pixY-max(np.abs(pixY-bboxPixY)), pixY+max(np.abs(pixY-bboxPixY))])

                    ############################################################3
                    #print('bboxPixX',bboxPixX)
                    #print('bboxPixY',bboxPixY)
                    print('checkpoint10 pass')
                    #############################################################

                    if np.any(bboxPixX <= 0) or np.any(bboxPixX > 640) or np.any(bboxPixY <= 0) or np.any(bboxPixY > 480) :
                        continue

                    p2_bboxRangePixels = np.array([[bboxPixX],[bboxPixY]])
                    p2_framePath = otherFramePrefix
                    p2_pixelCoords = [pixX,pixY]
                    p2_camCoords = ptCam
                    p2_camK = camK

                    p2 = {'bboxCornersCam': p2_bboxCornersCam, 'bboxRangePixels': p2_bboxRangePixels, 'framePath': p2_framePath,
                        'pixelCoords': p2_pixelCoords, 'camCoords': p2_camCoords, 'camK': p2_camK}
                    
                    corresFound = 1

                    break
        if corresFound == 0 :
            continue

        for otherFrameIdx in range(1,len(randFrameIdx)):
            randIdx = np.random.randint(1, len(randFrameIdx), size=1)
            otherFramePrefix = sceneDataList[randSceneIdx]['frameList'][randFrameIdx[int(randIdx)]]

            #########################################################
            otherFramePrefix = framePrefix = '/home/drzadmin/Desktop/3DMatch-pytorch/data/target/Cherry/seq-01/frame-000000'
            #########################################################

            p3_framePath = otherFramePrefix

            depthIm = Image.open(otherFramePrefix + '.depth.png')
            depthIm = np.array(depthIm)/1000       
            depthIm[depthIm > 6] = 0
            randDepthInd = np.random.randint(len(np.nonzero(depthIm)[0]))
            pixY,pixX = np.nonzero(depthIm)[0][randDepthInd], np.nonzero(depthIm)[1][randDepthInd]
            
            pixY = 248
            pixX = 302

            p3_pixelCoords = [pixX,pixY]

            ptCamZ = depthIm[pixY,pixX]
            pixY,pixX = pixY+1,pixX+1
            ptCamX = np.float((pixX-0.5-camK[0,2])*ptCamZ / camK[0,0])
            ptCamY = np.float((pixY-0.5-camK[1,2])*ptCamZ / camK[1,1])
            ptCam = np.array([[ptCamX],[ptCamY],[ptCamZ]])           

            p3_camCoords = ptCam

            with open(otherFramePrefix + '.pose.txt', 'r') as f:
                f = f.readlines()
            extCam2World = []
            for line in f:           
                line = line.split()
                extCam2World.append(line)
       
            extCam2World = array(extCam2World).astype(float)           

            p3World = np.dot(extCam2World[0:3,0:3],ptCam) + np.reshape(extCam2World[0:3,3],(3,1))            

            ##############################################3
            #print('p3World',p3World)
            print('checkpoint11 pass')
            ###################################################

            # Compute bounding box in pixel coordinates
            bboxRange = np.array([[ptCam[0]-voxelGridPatchRadius*voxelSize,ptCam[0]+voxelGridPatchRadius*voxelSize], 
                [ptCam[1]-voxelGridPatchRadius*voxelSize,ptCam[1]+voxelGridPatchRadius*voxelSize],
                [ptCam[2]-voxelGridPatchRadius*voxelSize,ptCam[2]+voxelGridPatchRadius*voxelSize]])
        
            bboxCorners = np.array([[bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,0],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1],bboxRange[0,1]],
                [bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1],bboxRange[1,0],bboxRange[1,0],bboxRange[1,1],bboxRange[1,1]],
                [bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1],bboxRange[2,0],bboxRange[2,1]]])
            
            bboxRange = np.reshape(bboxRange,(3,2))
            bboxCorners = np.reshape(bboxCorners,(3,8))

            ##########################################################3
            #print('bboxRange',bboxRange)
            #print('bboxCorners',bboxCorners)
            print('checkpoint12 pass')
            #############################################################

            p3_bboxCornersCam = bboxCorners
            bboxPixX = np.round((bboxCorners[0,:]*camK[0,0]/bboxCorners[2,:])+camK[0,2])
            bboxPixY = np.round((bboxCorners[1,:]*camK[1,1]/bboxCorners[2,:])+camK[1,2])

            bboxPixX = np.array([pixX-max(np.abs(pixX-bboxPixX)), pixX+max(np.abs(pixX-bboxPixX))])
            bboxPixY = np.array([pixY-max(np.abs(pixY-bboxPixY)), pixY+max(np.abs(pixY-bboxPixY))])

            ################################################################
            #print('bboxPixX',bboxPixX)
            #print('bboxPixY',bboxPixY)
            print('checkpoint13 pass')
            #################################################################
            if np.any(bboxPixX <= 0) or np.any(bboxPixX > 640) or np.any(bboxPixY <= 0) or np.any(bboxPixY > 480) :
                continue

            p3_bboxRangePixels = np.array([[bboxPixX],[bboxPixY]])
            p3_camK = camK

            # Check if 3D point is far enough to be considered a nonmatch
            if math.sqrt(np.sum((p1World-p3World)**2)) > -0.0001 :   ####################debug!!!!!!  > 0.1 ##########
                nonMatchFound = 1
                p3 = {'bboxCornersCam': p3_bboxCornersCam, 'bboxRangePixels': p3_bboxRangePixels, 'framePath': p3_framePath,
                        'pixelCoords': p3_pixelCoords, 'camCoords': p3_camCoords, 'camK': p3_camK}
                    
                break


    p1_voxelGridTDF = getPatchData(p1,voxelGridPatchRadius,voxelSize,voxelMargin)
    p2_voxelGridTDF = getPatchData(p2,voxelGridPatchRadius,voxelSize,voxelMargin)
    p3_voxelGridTDF = getPatchData(p3,voxelGridPatchRadius,voxelSize,voxelMargin)
    return p1_voxelGridTDF, p2_voxelGridTDF, p3_voxelGridTDF

def getPatchData(pointData,voxelGridPatchRadius,voxelSize,voxelMargin):
    depthIm = Image.open(pointData['framePath'] + '.depth.png')
    depthIm = np.array(depthIm)/1000
    depthIm[depthIm > 6] = 0

    lowBoundX = int(np.min(pointData['bboxRangePixels'][1,:]))
    highBoundX = int(np.max(pointData['bboxRangePixels'][1,:]))+1
    lowBoundY = int(np.min(pointData['bboxRangePixels'][0,:]))
    highBoundY = int(np.max(pointData['bboxRangePixels'][0,:]))+1

    depthPatch = depthIm[lowBoundX-1:highBoundX-1, lowBoundY-1:highBoundY-1]

    ##########################################################
    #print(np.shape(depthPatch))
    print('checkpoint14 pass')
    ##########################################################

    # Get TDF voxel grid local patches
    [pixX,pixY] = np.mgrid[lowBoundY:highBoundY, lowBoundX:highBoundX]
    pixX, pixY = pixX.T, pixY.T
    ##########################################################
    #print('pixX',pixX)
    #print('pixY',pixY)
    print('checkpoint15 pass')
    ######################################################3

    camX = np.array((pixX-pointData['camK'][0,2])*depthPatch/pointData['camK'][0,0])
    camY = np.array((pixY-pointData['camK'][1,2])*depthPatch/pointData['camK'][1,1])
    camZ = np.array(depthPatch)
    ####################################################
    #print(camX,camY,camZ)
    print('checkpoint16 pass')
    ###################################################

    ValidX,ValidY = np.nonzero(depthPatch)[0], np.nonzero(depthPatch)[1]
    ValidDepth = ValidX*np.shape(depthPatch)[1] + ValidY
    
    #################################################
    #print(ValidX,ValidY)
    #print(ValidDepth)
    print('checkpoint17 pass')
    #################################################

    camPts = np.append(np.reshape(camX.T,(np.size(camX),1)),np.reshape(camY.T,(np.size(camY),1)),1)
    camPts = np.append(camPts,np.reshape(camZ.T,(np.size(camZ),1)),1)
    #camPts = np.array([camX,camY,camZ])
    #camPts = np.reshape(camPts,(np.size(depthPatch),3))
    camPts = camPts[ValidDepth,:]

    ##################################################
    #print(np.reshape(camX.T,(np.size(camX),1)))
    #print(camPts)
    print('checkpoint18 pass')
    ##################################################

    #gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[(pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize, 
    #	np.float16(pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2):np.float16(pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize,
    #	(pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize]

    lowBoundX = pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2
    highBoundX = pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
    lowBoundY = pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2
    highBoundY = pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
    lowBoundZ = pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2
    highBoundZ = pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1

    gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[lowBoundX:highBoundX:voxelSize, lowBoundY:highBoundY:voxelSize, lowBoundZ:highBoundZ:voxelSize]

    gridPtsCam = np.append(np.reshape(gridPtsCamX.T,(np.size(gridPtsCamX),1)),np.reshape(gridPtsCamY.T,(np.size(gridPtsCamX),1)),1)
    gridPtsCam = np.append(gridPtsCam,np.reshape(gridPtsCamZ.T,(np.size(gridPtsCamZ),1)),1)

    ############################################################
    #print(lowBoundX,highBoundX,lowBoundY,highBoundY,lowBoundZ,highBoundZ)
    #print(np.shape(gridPtsCam))
    #print('gridPtsCam',gridPtsCam[34:39,:])
    #print('checkpoint19 pass')
    print('checkpoint19 pass')
    ##############################################################
    
    # Use 1-NN search to get TDF values
    #knnDist,knnIdx = spatial.KDTree(camPts).query(gridPtsCam)
    tree = cKDTree(camPts, leafsize=camPts.shape[0]+1)
    knnDist,knnIdx = tree.query(gridPtsCam, k=1, n_jobs=-1)
    TDFValues = knnDist/voxelMargin # truncate
    TDFValues[TDFValues > 1] = 1
    TDFValues = 1-TDFValues  # flip

    H = np.shape(gridPtsCamX)[0]
    W = np.shape(gridPtsCamX)[1]
    D = np.shape(gridPtsCamX)[2]

    voxelGridTDF = np.reshape(TDFValues.T,(H,W,D)).T.reshape((1,H,W,D))
    #print(voxelGridTDF[0,29,26])
    #voxelGridTDF = np.reshape(TDFValues,(None,H,W,D))

    ##############################################################
    #print(np.shape(gridPtsCamX))
    #print(np.shape(voxelGridTDF))

    print(voxelGridTDF[0,0,29,25])
    print(np.shape(voxelGridTDF))
    ################################################################
    return voxelGridTDF


  
#def data_loader(dataPath, batch_size):
def main():
	########################################################
    #dataPath = 'C:/Users/yifew/Desktop/DRZ/data'
    dataPath = '/home/drzadmin/Desktop/3DMatch-pytorch/data/target'
    trainScenes = ['Cherry']
    batch_size = 64
    ############################################################
    #trainScenes = ['Cherry', 'ChineseBanyan(0.25)', 'GiantRedwood', 'ItalianStonePine(0.3)',
    #	'KoreanStewartia', 'MountainMaple(0.3)', 'RedOak', 'WesternJuniper']

    # Local TDF voxel grid parameters
    voxelGridPatchRadius = 15  # in voxels
    voxelSize = 0.01  # in meters
    voxelMargin = voxelSize * 5
    # Make training scene frame lists
    sceneDataList = []

    for sceneIdx in range(len(trainScenes)):
        frameList = []
        scenePath = str(dataPath) + '/' + str(trainScenes[sceneIdx])
        os.chdir(scenePath)
        seqDir = glob.glob('seq-*')
        for seqIdx in range(len(seqDir)):
            seqName = seqDir[seqIdx]
            os.chdir(str(scenePath) + '/' + str(seqName))
            frameDir = glob.glob('frame-*.depth.png')
            #frameList = [None] * (len(frameDir))
            for frameIdx in range(len(frameDir)):
                framePath = os.path.join(scenePath,seqName,frameDir[frameIdx][0:-10])
                frameList.append(framePath)
        camKDir = str(scenePath) + '/' + 'camera-intrinsics.txt'
        with open(camKDir, 'r') as f:
            f = f.readlines()
        camK = []
        for line in f:
            line = line.split()
            camK.append(line)
        
        camK = array(camK).astype(float)
        
        sceneDataList.append({'name': trainScenes[sceneIdx], 'frameList': frameList, 'camK': camK})
        # Generate training correspondences
    batchDataP1 = np.empty(batch_size * 1 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2).reshape((batch_size, 1, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2))
    batchDataP2 = np.empty(batch_size * 1 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2).reshape((batch_size, 1, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2))
    batchDataP3 = np.empty(batch_size * 1 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2).reshape((batch_size, 1, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2))
    
    maxTries = 100
    NumSample = 0
        
    while(NumSample < batch_size):
        p1, p2, p3 = getPair(sceneDataList, maxTries, voxelGridPatchRadius, voxelSize, voxelMargin)
    
        batchDataP1[NumSample,:,:,:,:] = p1[:,:,:,:]
        batchDataP2[NumSample,:,:,:,:] = p2[:,:,:,:]
        batchDataP3[NumSample,:,:,:,:] = p3[:,:,:,:]
        
        NumSample = NumSample + 1
    return batchDataP1, batchDataP2, batchDataP3

if __name__ == '__main__':
    main()