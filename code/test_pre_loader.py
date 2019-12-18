import h5py
import numpy as np

len_cut = 5000

def test_loader(testfile, window_size):

    f = h5py.File(testfile)
    data = f['data']
    labels = f['labels']

    label = np.zeros((data.shape[1], f[labels[0,0]][:].shape[1]))
    data_p1 = np.zeros((data.shape[1], window_size, window_size, window_size))
    data_p2 = np.zeros((data.shape[1], window_size, window_size, window_size))

    for i in range(data.shape[1]):
        
        label[i] = f[labels[0,i]][:]
        data_p1[i] = f[f['data'][0,i]]['voxelGridTDF'][:]#.reshape(1,window_size,window_size,window_size,1)
        data_p2[i] = f[f['data'][1,i]]['voxelGridTDF'][:]#.reshape(1,window_size,window_size,window_size,1)

    print(testfile)
    print(label.shape)
    print(data_p1.shape)

    index0 = np.where(label[:,0] == 0)
    index1 = np.where(label[:,0] == 1)

    return data_p1, data_p2, label

def cutter(input1, input2, input3, len):

	idx = np.random.randint(0, input1.shape[0]-1, len)

	return input1[idx,:], input2[idx,:], input3[idx,:]


print('load 30 frame data...................')

print('...................trunk_test')
data_p1_trunk, data_p2_trunk, label_trunk = test_loader('trunk_test.mat', 30)
data_p1_trunk, data_p2_trunk, label_trunk = cutter(data_p1_trunk, data_p2_trunk, label_trunk, len_cut)

print('saving 1/8')
np.save('trunk1_big.npy', data_p1_trunk)
np.save('trunk2_big.npy', data_p2_trunk)
np.save('trunk3_big.npy', label_trunk)

data_p1_trunk, data_p2_trunk, label_trunk = [], [], []

print('...................leaves_test')
data_p1_leaves, data_p2_leaves, label_leaves = test_loader('leaves_test.mat', 30)
data_p1_leaves, data_p2_leaves, label_leaves = cutter(data_p1_leaves, data_p2_leaves, label_leaves, len_cut)

print('saving 2/8')
np.save('leaves1_big.npy', data_p1_leaves)
np.save('leaves2_big.npy', data_p2_leaves)
np.save('leaves3_big.npy', label_leaves)

data_p1_leaves, data_p2_leaves, label_leaves = [], [], []

print('...................branch_test')
data_p1_branch, data_p2_branch, label_branch = test_loader('branch_test.mat', 30)
data_p1_branch, data_p2_branch, label_branch = cutter(data_p1_branch, data_p2_branch, label_branch, len_cut)

print('saving 3/8')
np.save('branches1_big.npy', data_p1_branch)
np.save('branches2_big.npy', data_p2_branch)
np.save('branches3_big.npy', label_branch)

data_p1_branch, data_p2_branch, label_branch = [], [], []

print('...................trunk_test')
data_p1_test, data_p2_test, label_test = test_loader('test.mat', 30)
data_p1_test, data_p2_test, label_test = cutter(data_p1_test, data_p2_test, label_test, len_cut)

print('saving 4/8')
np.save('test1_big.npy', data_p1_test)
np.save('test2_big.npy', data_p2_test)
np.save('test3_big.npy', label_test)

data_p1_test, data_p2_test, label_test = [], [], []


print('load 45 frame data...................')

print('...................trunk_test_45')
data_p1_trunk_45, data_p2_trunk_45, label_trunk_45 = test_loader('trunk_test_45.mat', 45)
data_p1_trunk_45, data_p2_trunk_45, label_trunk_45 = cutter(data_p1_trunk_45, data_p2_trunk_45, label_trunk_45, len_cut)

print('saving 5/8')
np.save('trunk1_45_big.npy', data_p1_trunk_45)
np.save('trunk2_45_big.npy', data_p2_trunk_45)
np.save('trunk3_45_big.npy', label_trunk_45)

data_p1_trunk_45, data_p2_trunk_45, label_trunk_45 = [], [], []

print('...................leaves_test_45')
data_p1_leaves_45, data_p2_leaves_45, label_leaves_45 = test_loader('leaves_test_45.mat', 45)
data_p1_leaves_45, data_p2_leaves_45, label_leaves_45 = cutter(data_p1_leaves_45, data_p2_leaves_45, label_leaves_45, len_cut)

print('saving 6/8')
np.save('leaves1_45_big.npy', data_p1_leaves_45)
np.save('leaves2_45_big.npy', data_p2_leaves_45)
np.save('leaves3_45_big.npy', label_leaves_45)

data_p1_leaves_45, data_p2_leaves_45, label_leaves_45 = [], [], []

print('...................branch_test_45')
data_p1_branch_45, data_p2_branch_45, label_branch_45 = test_loader('branch_test_45.mat', 45)
data_p1_branch_45, data_p2_branch_45, label_branch_45 = cutter(data_p1_branch_45, data_p2_branch_45, label_branch_45, len_cut)

print('saving 7/8')
np.save('branches1_45_big.npy', data_p1_branch_45)
np.save('branches2_45_big.npy', data_p2_branch_45)
np.save('branches3_45_big.npy', label_branch_45)

data_p1_branch_45, data_p2_branch_45, label_branch_45 = [], [], []

print('...................test_45')
data_p1_test_45, data_p2_test_45, label_test_45 = test_loader('test_45.mat', 45)
data_p1_test_45, data_p2_test_45, label_test_45 = cutter(data_p1_test_45, data_p2_test_45, label_test_45, len_cut)

print('saving 8/8')
np.save('test1_45_big.npy', data_p1_test_45)
np.save('test2_45_big.npy', data_p2_test_45)
np.save('test3_45_big.npy', label_test_45)

data_p1_test_45, data_p2_test_45, label_test_45 = [], [], []



