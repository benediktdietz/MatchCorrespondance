from numpy import array
import numpy as np
import os
import glob

dataPath = '/home/drzadmin/Desktop/3DMatch-pytorch/data'
batch_size = 64

sceneDataList = {}
sceneDataList['sceneName'] = {}

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

print(sceneDataList['sceneName']['7-scenes-stairs']['seqList']['seq-01'][array([0,1])])
    

