import argparse
import os
import shutil
import time
from numpy import array
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from loadTrainDataGPU2 import data_loader
from loadTrainDataGPU2 import datalist_loader
from loadDADataGPU import targetlist_loader
#from test_dataloader2 import data_loader
#from test_dataloader2 import datalist_loader

from getError import ErrorRate95Recall

parser = argparse.ArgumentParser(description='PyTorch 3DMatch Training')
parser.add_argument('--data', metavar='DIR',help='path to dataset')
parser.add_argument('--iter', default=10000, type=int, metavar='N', help='number of total iterations to run')
parser.add_argument('--test-interval', default=100, type=int, metavar='N', help='number of iterations of interval between each test')
parser.add_argument('--start-iter', default=0, type=int, metavar='N', help='manual iterations number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--step-size', default=4000, type=int, metavar='N', help='number of iterations to set lr to 0.1*current_lr')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', '-s', default=100, type=int,metavar='N', help='save checkpoints (default: 1000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='test on test set')

class Match3DNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Match3DNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(64),
            
            # conv2
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(64),
            
            # maxpool
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1),
            # conv3
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(128),
            
            # conv4
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(128),

            nn.MaxPool3d(kernel_size=2, stride=2, padding=1),
            
            # conv5
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(256),
            
            # conv6
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(256),
            
            # conv7
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            #nn.BatchNorm3d(512),
            
            # conv8
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm3d(512),
            nn.ReLU(inplace=True), 
            #nn.BatchNorm3d(512),       
            )

    def forward_once(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.

        forward one type of the input
        """
        x = self.features(x)
        x = x.view(x.size(0), 512) # reshpe it into (batch_size, feature_dimention)
        return x

    def forward(self, p1, p2, p3):
        """
        forward three times to compute the features of matched pair (p1,p2) and nonmatch pair (p1,p3)
        """
        p1 = self.forward_once(p1)
        p2 = self.forward_once(p2)
        p3 = self.forward_once(p3)
        return p1, p2, p3


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)

        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def main():
    global args
    args = parser.parse_args()

    # create model
    model = Match3DNet()
    model.apply(weights_init)

    # define loss function (criterion) and optimizer
    criterion = ContrastiveLoss().cuda()  #l2
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # optionally resume from a checkpoint
    #print(os.path.isfile('/old/xavier-batch32-lr0.001/checkpoint_5000.pth'))
    if args.resume:
        #if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        #else:
        #    print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    window_size = 45
    traindir = args.data + '/data/target'
    testdir_indoor = args.data + '/test/indoor_test/test'
    testdir_tree = args.data + '/test/tree_test_window_size_45/test_45'
    testdir_trunk = args.data + '/test/tree_test_window_size_45/trunk_test_45'
    testdir_branch = args.data + '/test/tree_test_window_size_45/branch_test_45'
    testdir_leaves = args.data + '/test/tree_test_window_size_45/leaves_test_45'

    sceneDataList, trainScenes = targetlist_loader(traindir)

    if args.test:
        batch_time, loss1, error_rate1 = test(testdir_tree, model, criterion)
        print('Test tree: \t'
            	  'Time {0:3f}\t'
                  'Loss {1:4f}\t'
                  'ErrorRate {2:3f}'.format(
                   batch_time, loss1, error_rate1))

        batch_time, loss2, error_rate2 = test(testdir_trunk, model, criterion)
        print('Test trunk: \t'
                  'Time {0:3f}\t'
                  'Loss {1:4f}\t'
                  'ErrorRate {2:3f}'.format(
                   batch_time, loss2, error_rate2))      

        batch_time, loss3, error_rate3 = test(testdir_branch, model, criterion)
        print('Test branch: \t'
                  'Time {0:3f}\t'
                  'Loss {1:4f}\t'
                  'ErrorRate {2:3f}'.format(
                   batch_time, loss3, error_rate3))

        batch_time, loss4, error_rate4 = test(testdir_leaves, model, criterion)
        print('Test leaves: \t'
                  'Time {0:3f}\t'
                  'Loss {1:4f}\t'
                  'ErrorRate {2:3f}'.format(
                   batch_time, loss4, error_rate4))      

        return

    print("=> Training network... ")

    for iter in range(args.start_iter, args.iter):
        
        lr = adjust_learning_rate(optimizer, iter)

        # test on test set
        if iter % args.test_interval == 0:
            batch_time, loss1, error_rate1 = test(testdir_tree, model, criterion)
            print('Test tree: \t'
                      'Time {0:3f}\t'
                      'Loss {1:4f}\t'
                      'ErrorRate {2:3f}'.format(
                       batch_time, loss1, error_rate1))

            batch_time, loss2, error_rate2 = test(testdir_trunk, model, criterion)
            print('Test trunk: \t'
                      'Time {0:3f}\t'
                      'Loss {1:4f}\t'
                      'ErrorRate {2:3f}'.format(
                       batch_time, loss2, error_rate2))      

            batch_time, loss3, error_rate3 = test(testdir_branch, model, criterion)
            print('Test branch: \t'
                      'Time {0:3f}\t'
                      'Loss {1:4f}\t'
                      'ErrorRate {2:3f}'.format(
                       batch_time, loss3, error_rate3))

            batch_time, loss4, error_rate4 = test(testdir_leaves, model, criterion)
            print('Test leaves: \t'
                      'Time {0:3f}\t'
                      'Loss {1:4f}\t'
                      'ErrorRate {2:3f}'.format(
                       batch_time, loss4, error_rate4))   
            with open('/home/drzadmin/Desktop/3DMatch-pytorch/log/testloss.txt', 'a') as out:
                out.write(str(iter) + ' tree ' + str(loss1) + ' ' + str(error_rate1) + '\n')
                out.write(str(iter) + ' trunk ' + str(loss2) + ' ' + str(error_rate2) + '\n')
                out.write(str(iter) + ' branch ' + str(loss3) + ' ' + str(error_rate3) + '\n')
                out.write(str(iter) + ' leaves ' + str(loss4) + ' ' + str(error_rate4) + '\n')

        # train for one iteration
        dataPath = traindir
        batch_time, data_time, mlosses, nonmlosses = train(sceneDataList, trainScenes, dataPath, args.batch_size, model, criterion, optimizer, iter)
            
        # print the loss according to print frequency
        if iter % args.print_freq == 0:
            print('Iteration: [{0}/{1}]\t'
                  'Time {2:3f}\t'
                  'Data time {3:3f}\t'
                  'learning rate {4:3f}\t'
                  'MatchLoss {5:4f}\t'
                  'NonMatchLoss {6:4f}\t'.format(
                   iter, args.iter, batch_time, data_time, lr, mlosses, nonmlosses))
            with open('/home/drzadmin/Desktop/3DMatch-pytorch/log/trainloss.txt', 'a') as out:
                out.write(str(iter) + ' ' + str(mlosses) + ' ' + str(nonmlosses) + '\n')
        
        # save checkpoint according to save frequency
        if iter % args.save_freq == 0:
            print("=> Saving checkpoint... ")
            torch.save({
                'iter': iter + 1,
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, '/home/drzadmin/Desktop/3DMatch-pytorch/weights/tree_from_scratch_window_size_45/checkpoint_tree_45_' + str(iter) +'.pth')
            print('=> Checkpoint saved')

def train(sceneDataList, trainScenes, dataPath, batch_size, model, criterion, optimizer, iteration):
    
    # switch to train mode
    model.train().cuda()
    end = time.time()
    p1, p2, p3 = data_loader(sceneDataList, trainScenes, dataPath, batch_size)
    
    #p1, p2, p3 = torch.FloatTensor(p1).cuda(), torch.FloatTensor(p2).cuda(), torch.FloatTensor(p3).cuda()

    # measure data loading time
    data_time = time.time() - end

    # create label
    label_match = torch.FloatTensor(np.ones(batch_size)).cuda()
    label_nonmatch = torch.FloatTensor(np.zeros(batch_size)).cuda()
    
    # compute features
    p1, p2, p3, label_match, label_nonmatch = torch.autograd.Variable(p1, requires_grad=True), torch.autograd.Variable(p2, requires_grad=True), torch.autograd.Variable(p3, requires_grad=True), torch.autograd.Variable(label_match, requires_grad=True), torch.autograd.Variable(label_nonmatch, requires_grad=True)
    
    feature_p1, feature_p2, feature_p3 = model(p1, p2, p3)
    
    # compute two losses: match loss and nonmatch loss
    mlosses = criterion(feature_p1, feature_p2, label_match)  
    nonmlosses = criterion(feature_p1, feature_p3, label_nonmatch)
    
    # compute gradient and do SGD step
    optimizer.zero_grad()
    losses = mlosses + nonmlosses
    losses.backward()    
    optimizer.step()

    # measure batch time
    batch_time = time.time() - end
    end = time.time()

    return batch_time, data_time, mlosses.data.cpu().numpy()[0], nonmlosses.data.cpu().numpy()[0]

def test(testdir, model, criterion):
    print("=> Testing network... ")
    # switch to test mode
    window_size = 45
    model.eval().cuda()

    end = time.time()
    
    # load test data
    f = h5py.File(testdir + '.mat')
    data = f['data']
    labels = f['labels']

    descDistsall = []
    loss_total = []

    for i in range(data.shape[1]):
        label = torch.Tensor(f[labels[0,i]][:])
        #label = label.cuda(async=True)

        data_p1 = torch.FloatTensor(f[f['data'][0,i]]['voxelGridTDF'][:].reshape(1,1,window_size,window_size,window_size))
        data_p2 = torch.FloatTensor(f[f['data'][1,i]]['voxelGridTDF'][:].reshape(1,1,window_size,window_size,window_size))

        label = torch.autograd.Variable(label.cuda(), volatile=True)
        data_p1 = torch.autograd.Variable(data_p1.cuda(), volatile=True)
        data_p2 = torch.autograd.Variable(data_p2.cuda(), volatile=True)

        # compute output
        DesP1, DesP2, _ = model(data_p1, data_p2, data_p2)

        loss = criterion(DesP1, DesP2, label)
        loss = loss.data.cpu().numpy()[0]

        loss_total.append(loss)

        DesP1, DesP2 = DesP1.data.cpu().numpy(), DesP2.data.cpu().numpy()
        descDists = np.sqrt(np.sum((DesP1 - DesP2)**2))

        descDistsall.append(descDists)
        # measure accuracy and record loss
    
    error_rate = ErrorRate95Recall(array(descDistsall), testdir)
    loss = np.mean(array(loss_total))

    # measure test time
    batch_time = time.time() - end
    end = time.time()

    return batch_time, loss, error_rate

def adjust_learning_rate(optimizer, iter):
    """Sets the learning rate to the initial LR decayed by 10 every step_size iteration"""
    lr = args.lr * (0.1 ** (iter // args.step_size))
    for param_group in optimizer.param_groups:
    	param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()