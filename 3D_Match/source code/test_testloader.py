import numpy as np
import h5py
from numpy import array
import torch

f = h5py.File('data/test_data.mat')
data = f['data']
print(data.shape[1])
#ref = test[0,0]
#data = f[f['data'][0,0]]['voxelGridTDF'][:]
#ref = array(data['voxelGridTDF'][:])

labels = f['labels']
descDistsall = torch.FloatTensor()
print(descDistsall)
#print(torch.FloatTensor(f[f['data'][0,0]]['voxelGridTDF'][:]))