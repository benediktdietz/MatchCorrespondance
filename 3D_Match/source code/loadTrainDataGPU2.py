import os
import glob
from numpy import array
from PIL import Image
import numpy as np
from numpy.linalg import inv
from scipy import spatial
from scipy.spatial import cKDTree
import math
import time
import torch

def getPair(sceneDataList, trainScenes, dataPath, maxTries, voxelGridPatchRadius, voxelSize, voxelMargin):
    corresFound = 0
    nonMatchFound = 0

    while(corresFound < 1 and nonMatchFound < 1):
        # Pick a random scene, random seq and a random frame
        randSceneIdx = np.random.randint(len(sceneDataList['sceneName']))
        randSeqIdx = np.random.randint(len(sceneDataList['sceneName'][trainScenes[randSceneIdx]]['seqList']))
        scenePath = str(dataPath) + '/' + str(trainScenes[randSceneIdx])
        os.chdir(scenePath)
        seqDir = glob.glob('seq-*')
        seqName = seqDir[randSeqIdx]

        randFrameIdx = np.random.randint(len(sceneDataList['sceneName'][trainScenes[randSceneIdx]]['seqList'][seqName]), size=maxTries)
        camK = sceneDataList['sceneName'][trainScenes[randSceneIdx]]['camK']

        # Find a random 3D point (in world coordinates) in a random frame
        framePrefix = sceneDataList['sceneName'][trainScenes[randSceneIdx]]['seqList'][seqName][randFrameIdx[0]]
   
        ##################################################
        #framePrefix = '/home/drzadmin/Desktop/3DMatch-pytorch/data/Cherry/seq-01/frame-000000'

        ###################################################
        p1_framePath = framePrefix
        depthIm = Image.open(framePrefix + '.depth.png')
        depthIm = np.array(depthIm)/1000       
        depthIm[depthIm > 6] = 0
        randDepthInd = np.random.randint(len(np.nonzero(depthIm)[0]))
        pixY,pixX = np.nonzero(depthIm)[0][randDepthInd], np.nonzero(depthIm)[1][randDepthInd]
        
        ####################################################
        #pixX,pixY = 302, 248

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
        #print('checkpoint1 pass')
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
        #print('checkpoint2 pass')
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
        #print('checkpoint3 pass')
        ##############################################################
        bboxPixX = np.round((bboxCorners[0,:]*camK[0,0]/bboxCorners[2,:])+camK[0,2])
        bboxPixY = np.round((bboxCorners[1,:]*camK[1,1]/bboxCorners[2,:])+camK[1,2])

        bboxPixX = np.array([pixX-max(np.abs(pixX-bboxPixX)), pixX+max(np.abs(pixX-bboxPixX))])
        bboxPixY = np.array([pixY-max(np.abs(pixY-bboxPixY)), pixY+max(np.abs(pixY-bboxPixY))])

        ############################################################
        #print('bboxPixX',bboxPixX)
        #print('bboxPixY',bboxPixY)
        #print('checkpoint4 pass')
        #############################################################
        
        if np.any(bboxPixX <= 0) or np.any(bboxPixX > 640) or np.any(bboxPixY <= 0) or np.any(bboxPixY > 480) :
            continue

        p1_bboxRangePixels = np.array([[bboxPixX],[bboxPixY]])
        p1_camK = camK
        p1 = {'bboxCornersCam': p1_bboxCornersCam, 'bboxRangePixels': p1_bboxRangePixels, 'framePath': p1_framePath,
                    'pixelCoords': p1_pixelCoords, 'camCoords': p1_camCoords, 'camK': p1_camK}

        # Loop through other random frames to find a positive point match
        for otherFrameIdx in range(1,len(randFrameIdx)):

            otherFramePrefix = sceneDataList['sceneName'][trainScenes[randSceneIdx]]['seqList'][seqName][randFrameIdx[otherFrameIdx]]
            ###################################################
            #otherFramePrefix = '/home/drzadmin/Desktop/3DMatch-pytorch/data/Cherry/seq-01/frame-000020'
            #####################################################

            #Check if 3D point is within camera view frustum
            with open(otherFramePrefix + '.pose.txt', 'r') as f:
                f = f.readlines()
            extCam2World = []
            for line in f:
                line = line.split()
                extCam2World.append(line)

            extCam2World = array(extCam2World).astype(float)
            
            if np.isnan(np.sum(extCam2World[:])):
                continue

            p2CamLoc = extCam2World[0:3,3]
            if np.sqrt(np.sum((p1CamLoc-p2CamLoc)**2)) < 1 :
                continue
            
            extWorld2Cam = inv(extCam2World)

            #########################################################
            #print('extWorld2Cam',extWorld2Cam)
            #print('checkpoint5 pass')
            ##########################################################

            ptCam = np.dot(extWorld2Cam[0:3,0:3],p1World) + np.reshape(extWorld2Cam[0:3,3],(3,1))

            ########################################################
            #print('ptCam',ptCam)
            #print('checkpoint6 pass')
            #########################################################


            pixX = np.round((ptCam[0]*camK[0,0]/ptCam[2])+camK[0,2]+0.5).astype(int)
            pixY = np.round((ptCam[1]*camK[1,1]/ptCam[2])+camK[1,2]+0.5).astype(int)

            #########################################################
            #print('pixX',pixX)
            #print('pixY',pixY)
            #print('checkpoint7 pass')
            #########################################################
            
            if pixX > 0 and pixX <= 640 and pixY > 0 and pixY <= 480 :
                depthIm = Image.open(otherFramePrefix + '.depth.png')
                depthIm = np.array(depthIm)/1000       
                depthIm[depthIm > 6] = 0
                
                if abs(depthIm[pixY-1,pixX-1]-ptCam[2]) < 0.03 :     ########################!!!!! debug !!!!!! change to < 0.03 after################
                    ptCamZ = depthIm[pixY-1,pixX-1]
                    ptCamX = float((pixX-0.5-camK[0,2])*ptCamZ/camK[0,0])
                    ptCamY = float((pixY-0.5-camK[1,2])*ptCamZ/camK[1,1])
                    ptCam = np.array([[ptCamX],[ptCamY],[ptCamZ]])

                    ######################################################3
                    #print('ptCam',ptCam)
                    #print('checkpoint8 pass')
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
                    #print('checkpoint9 pass')
                    #############################################################

                    p2_bboxCornersCam = bboxCorners
                    bboxPixX = np.round((bboxCorners[0,:]*camK[0,0]/bboxCorners[2,:])+camK[0,2])
                    bboxPixY = np.round((bboxCorners[1,:]*camK[1,1]/bboxCorners[2,:])+camK[1,2])

                    bboxPixX = np.array([pixX-max(np.abs(pixX-bboxPixX)), pixX+max(np.abs(pixX-bboxPixX))])
                    bboxPixY = np.array([pixY-max(np.abs(pixY-bboxPixY)), pixY+max(np.abs(pixY-bboxPixY))])

                    ############################################################3
                    #print('bboxPixX',bboxPixX)
                    #print('bboxPixY',bboxPixY)
                    #print('checkpoint10 pass')
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
            
            otherFramePrefix = sceneDataList['sceneName'][trainScenes[randSceneIdx]]['seqList'][seqName][randFrameIdx[int(randIdx)]]
            #########################################################
            #otherFramePrefix = framePrefix = '/home/drzadmin/Desktop/3DMatch-pytorch/data/Cherry/seq-01/frame-000000'
            #########################################################

            p3_framePath = otherFramePrefix

            depthIm = Image.open(otherFramePrefix + '.depth.png')
            depthIm = np.array(depthIm)/1000       
            depthIm[depthIm > 6] = 0
            randDepthInd = np.random.randint(len(np.nonzero(depthIm)[0]))
            pixY,pixX = np.nonzero(depthIm)[0][randDepthInd], np.nonzero(depthIm)[1][randDepthInd]
            
            #########################################################
            #pixY = 248
            #pixX = 302
            #########################################################

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
            #print('checkpoint11 pass')
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
            #print('checkpoint12 pass')
            #############################################################

            p3_bboxCornersCam = bboxCorners
            bboxPixX = np.round((bboxCorners[0,:]*camK[0,0]/bboxCorners[2,:])+camK[0,2])
            bboxPixY = np.round((bboxCorners[1,:]*camK[1,1]/bboxCorners[2,:])+camK[1,2])

            bboxPixX = np.array([pixX-max(np.abs(pixX-bboxPixX)), pixX+max(np.abs(pixX-bboxPixX))])
            bboxPixY = np.array([pixY-max(np.abs(pixY-bboxPixY)), pixY+max(np.abs(pixY-bboxPixY))])

            ################################################################
            #print('bboxPixX',bboxPixX)
            #print('bboxPixY',bboxPixY)
            #print('checkpoint13 pass')
            #################################################################
            if np.any(bboxPixX <= 0) or np.any(bboxPixX > 640) or np.any(bboxPixY <= 0) or np.any(bboxPixY > 480) :
                continue

            p3_bboxRangePixels = np.array([[bboxPixX],[bboxPixY]])
            p3_camK = camK

            # Check if 3D point is far enough to be considered a nonmatch
            if math.sqrt(np.sum((p1World-p3World)**2)) > 0.1 :   ####################debug!!!!!!  > 0.1 ##########
                nonMatchFound = 1
                p3 = {'bboxCornersCam': p3_bboxCornersCam, 'bboxRangePixels': p3_bboxRangePixels, 'framePath': p3_framePath,
                        'pixelCoords': p3_pixelCoords, 'camCoords': p3_camCoords, 'camK': p3_camK}
                    
                break

    #end = time.time()
    p1_voxelGridTDF = getPatchData(p1,voxelGridPatchRadius,voxelSize,voxelMargin)
    p2_voxelGridTDF = getPatchData(p2,voxelGridPatchRadius,voxelSize,voxelMargin)
    p3_voxelGridTDF = getPatchData(p3,voxelGridPatchRadius,voxelSize,voxelMargin)
    #print(time.time()-end)
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
    #print('checkpoint14 pass')
    ##########################################################

    # Get TDF voxel grid local patches
    [pixX,pixY] = np.mgrid[lowBoundY:highBoundY, lowBoundX:highBoundX]
    pixX, pixY = pixX.T, pixY.T
    ##########################################################
    #print(lowBoundY-highBoundY)
    #print('pixX',np.size(pixX))
    #print('pixY',pixY)
    #print('checkpoint15 pass')
    ######################################################3

    if np.size(pixX)>60000:
        camX = np.matrix.round(array((pixX-pointData['camK'][0,2])*depthPatch/pointData['camK'][0,0]/5),3)*5
        camY = np.matrix.round(array((pixY-pointData['camK'][1,2])*depthPatch/pointData['camK'][1,1]/5),3)*5
        camZ = np.matrix.round(array(depthPatch/5),3)*5
    else:
        camX = array((pixX-pointData['camK'][0,2])*depthPatch/pointData['camK'][0,0])
        camY = array((pixY-pointData['camK'][1,2])*depthPatch/pointData['camK'][1,1])
        camZ = array(depthPatch)
    ####################################################
    #print(np.shape(camZ))
    #print(camX,camY,camZ)
    #print('checkpoint16 pass')
    ###################################################

    ValidX,ValidY = np.nonzero(depthPatch)[0], np.nonzero(depthPatch)[1]
    ValidDepth = ValidX*np.shape(depthPatch)[1] + ValidY
    
    #################################################
    #print(ValidX,ValidY)
    #print('ValidDepth',np.shape(ValidDepth))
    #print('checkpoint17 pass')
    #################################################

    camPts = np.append(np.reshape(camX.T,(np.size(camX),1)),np.reshape(camY.T,(np.size(camY),1)),1)
    camPts = np.append(camPts,np.reshape(camZ.T,(np.size(camZ),1)),1)
    #camPts = np.array([camX,camY,camZ])
    #camPts = np.reshape(camPts,(np.size(depthPatch),3))
    camPts = camPts[ValidDepth,:]
    camPts = np.unique(camPts, axis=0) ###########################3

    n1 = torch.FloatTensor(camPts).cuda()

    ##################################################
    #print(np.reshape(camX.T,(np.size(camX),1)))
    #print(camPts)
    #print('checkpoint18 pass')
    ##################################################

    #gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[(pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize, 
    #   np.float16(pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2):np.float16(pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize,
    #   (pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2):(pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2):voxelSize]

    lowBoundX = pointData['camCoords'][0]-voxelGridPatchRadius*voxelSize+voxelSize/2
    highBoundX = pointData['camCoords'][0]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
    lowBoundY = pointData['camCoords'][1]-voxelGridPatchRadius*voxelSize+voxelSize/2
    highBoundY = pointData['camCoords'][1]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1
    lowBoundZ = pointData['camCoords'][2]-voxelGridPatchRadius*voxelSize+voxelSize/2
    highBoundZ = pointData['camCoords'][2]+voxelGridPatchRadius*voxelSize-voxelSize/2 + voxelSize*0.1

    gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[lowBoundX:highBoundX:voxelSize, lowBoundY:highBoundY:voxelSize, lowBoundZ:highBoundZ:voxelSize]

    gridPtsCam = np.append(np.reshape(gridPtsCamX.T,(np.size(gridPtsCamX),1)),np.reshape(gridPtsCamY.T,(np.size(gridPtsCamX),1)),1)
    gridPtsCam = np.append(gridPtsCam,np.reshape(gridPtsCamZ.T,(np.size(gridPtsCamZ),1)),1)

    n2_1 = torch.FloatTensor(gridPtsCam[:4556,:]).cuda()
    n2_2 = torch.FloatTensor(gridPtsCam[4556:9113,:]).cuda()
    n2_3 = torch.FloatTensor(gridPtsCam[9113:13669,:]).cuda()
    n2_4 = torch.FloatTensor(gridPtsCam[13669:18225,:]).cuda()
    n2_5 = torch.FloatTensor(gridPtsCam[18225:22781,:]).cuda()
    n2_6 = torch.FloatTensor(gridPtsCam[22781:27338,:]).cuda()
    n2_7 = torch.FloatTensor(gridPtsCam[27338:31886,:]).cuda()
    n2_8 = torch.FloatTensor(gridPtsCam[31886:36450,:]).cuda()
    n2_9 = torch.FloatTensor(gridPtsCam[36450:41006,:]).cuda()
    n2_10 = torch.FloatTensor(gridPtsCam[41006:45563,:]).cuda()
    n2_11 = torch.FloatTensor(gridPtsCam[45563:50119,:]).cuda()
    n2_12 = torch.FloatTensor(gridPtsCam[50119:54675,:]).cuda()
    n2_13 = torch.FloatTensor(gridPtsCam[54675:59231,:]).cuda()
    n2_14 = torch.FloatTensor(gridPtsCam[59231:63788,:]).cuda()
    n2_15 = torch.FloatTensor(gridPtsCam[63788:68344,:]).cuda()
    n2_16 = torch.FloatTensor(gridPtsCam[68344:72900,:]).cuda()
    n2_17 = torch.FloatTensor(gridPtsCam[72900:77456,:]).cuda()
    n2_18 = torch.FloatTensor(gridPtsCam[77456:82013,:]).cuda()
    n2_19 = torch.FloatTensor(gridPtsCam[82013:86569,:]).cuda()
    n2_20 = torch.FloatTensor(gridPtsCam[86569:,:]).cuda()

    ############################################################
    #print(lowBoundX,highBoundX,lowBoundY,highBoundY,lowBoundZ,highBoundZ)
    #print(np.shape(gridPtsCam))
    #print('gridPtsCam',gridPtsCam[34:39,:])
    #print('checkpoint19 pass')
    #print('checkpoint19 pass')
    ##############################################################

    #####################################################3
    '''
    with open(pointData['framePath'] + '.pose.txt', 'r') as f:
        f = f.readlines()
    extCam2World = []
    for line in f:
        line = line.split()
        extCam2World.append(line)

    extCam2World = array(extCam2World).astype(float)
    Camworld = (np.dot(extCam2World[0:3,0:3],camPts.T) + np.reshape(extCam2World[0:3,3],(3,1))).astype(int).T

    Camworld = np.unique(Camworld, axis=0)
    '''
    ########################################################
    
    # Use 1-NN search to get TDF values
    #knnDist,knnIdx = spatial.KDTree(camPts).query(gridPtsCam)
    
    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_1.size()[0],1))+torch.sum(n2_1**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_1)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist1 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_2.size()[0],1))+torch.sum(n2_2**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_2)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist2 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_3.size()[0],1))+torch.sum(n2_3**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_3)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist3 = torch.sqrt(knnDist)
    
    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_4.size()[0],1))+torch.sum(n2_4**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_4)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist4 = torch.sqrt(knnDist)
    
    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_5.size()[0],1))+torch.sum(n2_5**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_5)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist5 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_6.size()[0],1))+torch.sum(n2_6**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_6)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist6 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_7.size()[0],1))+torch.sum(n2_7**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_7)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist7 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_8.size()[0],1))+torch.sum(n2_8**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_8)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist8 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_9.size()[0],1))+torch.sum(n2_9**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_9)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist9 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_10.size()[0],1))+torch.sum(n2_10**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_10)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist10 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_11.size()[0],1))+torch.sum(n2_11**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_11)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist11 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_12.size()[0],1))+torch.sum(n2_12**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_12)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist12 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_13.size()[0],1))+torch.sum(n2_13**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_13)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist13 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_14.size()[0],1))+torch.sum(n2_14**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_14)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist14 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_15.size()[0],1))+torch.sum(n2_15**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_15)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist15 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_16.size()[0],1))+torch.sum(n2_16**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_16)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist16 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_17.size()[0],1))+torch.sum(n2_17**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_17)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist17 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_18.size()[0],1))+torch.sum(n2_18**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_18)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist18 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_19.size()[0],1))+torch.sum(n2_19**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_19)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist19 = torch.sqrt(knnDist)

    knnDist,_ = torch.min(torch.addmm(1.0,torch.t(torch.sum(n1**2, 1).repeat(n2_20.size()[0],1))+torch.sum(n2_20**2, 1).repeat(n1.size()[0],1),-2.0,n1,torch.t(n2_20)),0)
    knnDist = torch.clamp(knnDist,min=0.0)
    knnDist20 = torch.sqrt(knnDist)

    knnDist = torch.cat((knnDist1,knnDist2,knnDist3,knnDist4,knnDist5,knnDist6,knnDist7,knnDist8,knnDist9,knnDist10,knnDist11,knnDist12,knnDist13,knnDist14,knnDist15,knnDist16,knnDist17,knnDist18,knnDist19,knnDist20))
    #tree = cKDTree(camPts, leafsize=camPts.shape[0]+1)

    #knnDist_tree,knnIdx = tree.query(gridPtsCam, k=1, n_jobs=-1)

    #print(np.sum((knnDist.cpu().numpy()-knnDist_tree)**2))
    
    #knnDist = torch.FloatTensor(knnDist)

    '''
    knnDist = mp.zeros(np.shape(gridPtsCam)[0],).cuda()

    for i in range(np.shape(gridPtsCam)[0]):
        query_pt_x = gridPtsCam[i,0]
        query_pt_y = gridPtsCam[i,1]
        query_pt_z = gridPtsCam[i,2]
        pt_cam_x = camPts[:,0]
        pt_cam_y = camPts[:,1]
        pt_cam_z = camPts[:,2]

        query_dist = np.sqrt((pt_cam_x - query_pt_x)**2 + (pt_cam_y - query_pt_y)**2 + (pt_cam_z - query_pt_z)**2)
        knnDist[i] = np.min(query_dist)           
    '''
    TDFValues = knnDist/voxelMargin # truncate

    #TDFValues = TDFValues.cpu().numpy()

    TDFValues[TDFValues > 1] = 1
    TDFValues = 1-TDFValues  # flip

    H = np.shape(gridPtsCamX)[0]
    W = np.shape(gridPtsCamX)[1]
    D = np.shape(gridPtsCamX)[2]

    TDFValues = TDFValues.view(1,H,W,D)

    #voxelGridTDF = np.reshape(TDFValues.T,(H,W,D)).T.reshape((1,H,W,D))
    #print(voxelGridTDF[0,0,29,26])
    #voxelGridTDF = np.reshape(TDFValues,(None,H,W,D))

    ##############################################################
    #print(np.shape(gridPtsCamX))
    #print(np.shape(voxelGridTDF))

    #print(voxelGridTDF[0,0,29,25])
    #print(np.shape(voxelGridTDF))
    ################################################################
    return TDFValues


  
def datalist_loader(dataPath):
    ########################################################
    #dataPath = 'C:/Users/yifew/Desktop/DRZ/data'
    #dataPath = '/home/drzadmin/Desktop/3DMatch-pytorch/data'
    #trainScenes = ['Cherry']
    #batch_size = 64
    ############################################################
    sceneDataList = {}
    sceneDataList['sceneName'] = {}
    #trainScenes = ['Cherry', 'ChineseBanyan(0.25)', 'GiantRedwood', 'ItalianStonePine(0.3)',
    #   'KoreanStewartia', 'MountainMaple(0.3)', 'RedOak', 'WesternJuniper']
    #trainScenes = ['Cherry']

    
    trainScenes = ['rgbd-scenes-v2-scene_03', 'rgbd-scenes-v2-scene_02', 'rgbd-scenes-v2-scene_01',
        'analysis-by-synthesis-office2-5b', 'analysis-by-synthesis-office2-5a', 'analysis-by-synthesis-apt2-luke',
        'analysis-by-synthesis-apt2-living', 'analysis-by-synthesis-apt2-kitchen', 'analysis-by-synthesis-apt2-bed',
        'analysis-by-synthesis-apt1-living', 'analysis-by-synthesis-apt1-kitchen', 'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika',
        'sun3d-hotel_sf-scan1', 'sun3d-brown_cs_2-brown_cs2', 'sun3d-brown_cogsci_1-brown_cogsci_1',
        'bundlefusion-office3', 'bundlefusion-office2', 'bundlefusion-office1',
        'bundlefusion-office0', 'bundlefusion-copyroom', 'bundlefusion-apt2',
        'bundlefusion-apt1', 'bundlefusion-apt0', 'rgbd-scenes-v2-scene_14',
        'rgbd-scenes-v2-scene_13', 'rgbd-scenes-v2-scene_12', 'rgbd-scenes-v2-scene_11',
        'rgbd-scenes-v2-scene_10', 'rgbd-scenes-v2-scene_09', 'rgbd-scenes-v2-scene_08',
        'rgbd-scenes-v2-scene_07', 'rgbd-scenes-v2-scene_06', 'rgbd-scenes-v2-scene_05',
        'rgbd-scenes-v2-scene_04', 'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika', 'sun3d-mit_76_417-76-417b',
        'sun3d-mit_46_ted_lab1-ted_lab_2', 'sun3d-mit_32_d507-d507_2', 'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika',
        'sun3d-harvard_c8-hv_c8_3', 'sun3d-harvard_c6-hv_c6_1', 'sun3d-harvard_c5-hv_c5_1',
        'sun3d-harvard_c3-hv_c3_1', 'sun3d-hotel_nips2012-nips_4', 'sun3d-harvard_c11-hv_c11_2',
        'sun3d-brown_cs_3-brown_cs3', 'sun3d-brown_bm_4-brown_bm_4', 'sun3d-brown_bm_1-brown_bm_1',
        '7-scenes-office', '7-scenes-fire', '7-scenes-stairs', '7-scenes-heads', '7-scenes-pumpkin', '7-scenes-chess']
    
    # Make training scene frame lists

    for sceneIdx in range(len(trainScenes)):

        scenePath = str(dataPath) + '/' + str(trainScenes[sceneIdx])

        camKDir = str(scenePath) + '/' + 'camera-intrinsics.txt'
        with open(camKDir, 'r') as f:
            f = f.readlines()
        camK = []
        for line in f:
            line = line.split()
            camK.append(line)

        camK = array(camK).astype(float)
    
        sceneDataList['sceneName'][trainScenes[sceneIdx]] = {}
        sceneDataList['sceneName'][trainScenes[sceneIdx]]['camK'] = camK

        os.chdir(scenePath)
        seqDir = glob.glob('seq-*')

        sceneDataList['sceneName'][trainScenes[sceneIdx]]['seqList'] = {}

        for seqIdx in range(len(seqDir)):
            seqName = seqDir[seqIdx]
            os.chdir(str(scenePath) + '/' + str(seqName))

            frameDir = glob.glob('frame-*.depth.png')
            frameList = []
            for frameIdx in range(len(frameDir)):
                framePath = os.path.join(scenePath,seqName,frameDir[frameIdx][0:-10])
                frameList.append(framePath)
            sceneDataList['sceneName'][trainScenes[sceneIdx]]['seqList'][seqName] = frameList
    return sceneDataList, trainScenes


def data_loader(sceneDataList, trainScenes, dataPath, batch_size):
    # Generate training correspondences
    # Local TDF voxel grid parameters
    voxelGridPatchRadius = 22.5  # in voxels
    voxelSize = 0.01  # in meters
    voxelMargin = voxelSize * 5
    window_size = 45

    #batchDataP1 = np.empty(batch_size * 1 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2).reshape((batch_size, 1, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2))
    #batchDataP2 = np.empty(batch_size * 1 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2).reshape((batch_size, 1, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2))
    #batchDataP3 = np.empty(batch_size * 1 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2 * voxelGridPatchRadius * 2).reshape((batch_size, 1, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2, voxelGridPatchRadius * 2))
    

    batchDataP1 = torch.FloatTensor(batch_size,1,window_size,window_size,window_size).cuda()
    batchDataP2 = torch.FloatTensor(batch_size,1,window_size,window_size,window_size).cuda()
    batchDataP3 = torch.FloatTensor(batch_size,1,window_size,window_size,window_size).cuda()

    maxTries = 100
    NumSample = 0
        
    while(NumSample < batch_size):
        p1, p2, p3 = getPair(sceneDataList, trainScenes, dataPath, maxTries, voxelGridPatchRadius, voxelSize, voxelMargin)
    
        batchDataP1[NumSample,:,:,:,:] = p1[:,:,:,:]
        batchDataP2[NumSample,:,:,:,:] = p2[:,:,:,:]
        batchDataP3[NumSample,:,:,:,:] = p3[:,:,:,:]
        
        NumSample = NumSample + 1
    return batchDataP1, batchDataP2, batchDataP3