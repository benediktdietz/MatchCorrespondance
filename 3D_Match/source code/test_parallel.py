from multiprocessing import Process
import sys
from loadTrainData import data_loader
from loadTrainData import datalist_loader
#from test_dataloader2 import data_loader
#from test_dataloader2 import datalist_loader

import time

def func1():
    global rocket
    print('start func1')
    end = time.time()
    traindir = '/home/drzadmin/Desktop/3DMatch-pytorch/data'
    dataPath = traindir
    batch_size = 32
    sceneDataList, trainScenes = datalist_loader(traindir)
    p1, p2, p3 = data_loader(sceneDataList, trainScenes, dataPath, batch_size)
    batch_time = time.time() - end
    print('time', batch_time)
    print('end func1')

if __name__=='__main__':
    
    p1= Process(target = func1())
    p1.start()
    p2 = Process(target = func1())
    p2.start()

    