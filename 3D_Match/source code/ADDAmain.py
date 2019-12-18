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

from loadDADataGPU import data_loader
from loadDADataGPU import sourcelist_loader
from loadDADataGPU import targetlist_loader

from getError import ErrorRate95Recall

parser = argparse.ArgumentParser(description='PyTorch 3DMatch Training')
parser.add_argument('--data', metavar='DIR',help='path to dataset')
parser.add_argument('--iter', default=20000, type=int, metavar='N', help='number of total iterations to run')
parser.add_argument('--test-interval', default=50, type=int, metavar='N', help='number of iterations of interval between each test')
parser.add_argument('--start-iter', default=0, type=int, metavar='N', help='manual iterations number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--step-size', default=20000, type=int, metavar='N', help='number of iterations to set lr to 0.1*current_lr')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', '-s', default=200, type=int,metavar='N', help='save checkpoints (default: 1000)')
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
            nn.MaxPool3d(kernel_size=2, stride=2),
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

    def forward(self, p1):
        """
        forward three times to compute the features of matched pair (p1,p2) and nonmatch pair (p1,p3)
        """
        p1 = self.forward_once(p1)
        
        return p1


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

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 2)
            )
        
    def forward(self, x):
        out = self.classifier(x)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def main():
    global args
    args = parser.parse_args()

    weightDecay = 2.5e-4
    betas = (0.5, 0.999) 

    # create model
    sourceCNN = Match3DNet()

    # optionally resume from a checkpoint
    pretrained_weight_source = args.data + 'weights/source/checkpoint_5200.pth'
    pretrained_weight_target = args.data + 'weights/target/checkpointADDA_400.pth'
    testdir = args.data + '/test/tree_test'
    
    print("=> loading checkpoint '{}'".format(pretrained_weight_source))
    checkpoint = torch.load(pretrained_weight_source)
    
    sourceCNN.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (iter {})".format(pretrained_weight_source, checkpoint['iter']))
   
    sourceCNN.eval().cuda()

    cudnn.benchmark = True

    targetCNN = Match3DNet()

    
    print("=> loading checkpoint '{}'".format(pretrained_weight_target))
    checkpoint = torch.load(pretrained_weight_target)
    
    targetCNN.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (iter {})".format(pretrained_weight_target, checkpoint['iter']))

    targetCNN.train().cuda()

    for param in sourceCNN.parameters():
        param.requires_grad = False

    D = Discriminator()
    D.apply(weights_init)
    if args.resume:
        checkpoint = torch.load(args.resume)
        D.load_state_dict(checkpoint['discriminator_dict'])
        args.start_iter = checkpoint['iter']

    D.train().cuda()

    #Doptimizor = torch.optim.Adam(D.parameters(), lr=args.lr*10, betas = betas, weight_decay= weightDecay)
    #TargetOptimizor = torch.optim.Adam(targetCNN.parameters(), lr=0, betas = betas, weight_decay= weightDeca
    Doptimizor = torch.optim.SGD(D.parameters(), args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    TargetOptimizor = torch.optim.SGD(targetCNN.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criteria = torch.nn.CrossEntropyLoss().cuda()
    criteria_loss = ContrastiveLoss().cuda()

    # Data loading code
    sourcedir = args.data + '/data/source'
    targetdir = args.data + '/data/target'

    sceneDataListSource, trainScenesSource = sourcelist_loader(sourcedir)
    sceneDataListTarget, trainScenesTarget = targetlist_loader(targetdir)

    if args.test:
        batch_time, loss, error_rate = test(testdir, targetCNN, criteria_loss)
        print('Test: \t'
            	  'Time {0:3f}\t'
                  'Loss {1:4f}\t'
                  'ErrorRate {2:3f}'.format(
                   batch_time, loss, error_rate))           
        return

    print("=> Training network... ")

    for iter in range(args.start_iter, args.iter):
        
        lr = adjust_learning_rate(Doptimizor, iter)
        lr = adjust_learning_rate(TargetOptimizor, iter)

        # train for one iteration

        # test on test set
        if iter % args.test_interval == 0:
            batch_time, loss, error_rate = test(testdir, targetCNN, criteria_loss)
            print('Test: \t'
            	  'Time {0:3f}\t'
                  'Loss {1:4f}\t'
                  'ErrorRate {2:3f}'.format(
                   batch_time, loss, error_rate))
            with open('/home/drzadmin/Desktop/3DMatch-pytorch/log/testloss.txt', 'a') as out:
                out.write(str(iter) + ' ' + str(loss) + ' ' + str(error_rate) + '\n')
            
        # save checkpoint according to save frequency
        if iter % args.save_freq == 0:
            print("=> Saving checkpoint... ")
            torch.save({
                'iter': iter + 1,
                'state_dict' : targetCNN.state_dict(),
                'discriminator_dict' : D.state_dict(),
                }, '/home/drzadmin/Desktop/3DMatch-pytorch/weights/target/checkpointADDA_' + str(iter) +'.pth')
            print('=> Checkpoint saved')

        batch_time, data_time, Dloss, GANloss = train(sceneDataListSource, trainScenesSource, sceneDataListTarget, trainScenesTarget, sourcedir, targetdir, args.batch_size,  sourceCNN, targetCNN, D, criteria, Doptimizor, TargetOptimizor)

        # print the loss according to print frequency
        if iter % args.print_freq == 0:
            print('Iteration: [{0}/{1}]\t'
                  'Time {2:3f}\t'
                  'Data time {3:3f}\t'
                  'Learning rate {4:3f}\t'
                  'Dloss {5:3f}\t'
                  'GANloss {6:4f}\t'.format(
                   iter, args.iter, batch_time, data_time, lr, Dloss, GANloss))
            with open('/home/drzadmin/Desktop/3DMatch-pytorch/log/trainloss.txt', 'a') as out:
                out.write(str(iter) + ' ' + str(GANloss) + '\n')

def train(sceneDataListSource, trainScenesSource, sceneDataListTarget, trainScenesTarget, sourcedir, targetdir, batch_size, sourceCNN, targetCNN, D, criteria, Doptimizor, TargetOptimizor):

    # switch to train mode
    #model.train().cuda()
    end = time.time()

    sourceTDF = data_loader(sceneDataListSource, trainScenesSource, sourcedir, batch_size)
    targetTDF = data_loader(sceneDataListTarget, trainScenesTarget, targetdir, batch_size)

    # measure data loading time
    data_time = time.time() - end

    for param in D.parameters():
        param.requires_grad = True

    # compute features
    sourceFeature = sourceCNN(torch.autograd.Variable(sourceTDF,requires_grad=True))
    targetFeature = targetCNN(torch.autograd.Variable(targetTDF,requires_grad=True))

    for param in targetCNN.parameters():
        param.requires_grad = False
    
    for param in sourceCNN.parameters():
        param.requires_grad = False
    # compute GANloss

    predictionOnSourceImagesForD = D(sourceFeature.detach())
    predictionOnTargetImagesForD = D(targetFeature.detach())

    predictionOnD = torch.cat((predictionOnSourceImagesForD, predictionOnTargetImagesForD), 0)

    sourceLabels = torch.zeros(batch_size, 1).long().squeeze().cuda()
    targetLabels = torch.ones(batch_size, 1).long().squeeze().cuda()

    domainLabel = torch.cat((sourceLabels, targetLabels), 0)

    Doptimizor.zero_grad()
    DError = criteria(predictionOnD, torch.autograd.Variable(domainLabel))

    DError.backward()

    Doptimizor.step()

    # Training Target:
    for param in targetCNN.parameters():
        param.requires_grad = True

    targetFeatureT = targetCNN(torch.autograd.Variable(targetTDF,requires_grad=True))
    targetLabelsT = torch.zeros(batch_size, 1).long().squeeze().cuda()

    predictionOnTargetImages = D(targetFeatureT)

    targetLabelsT = torch.autograd.Variable(targetLabelsT)

    TargetOptimizor.zero_grad()

    for param in D.parameters():
        param.requires_grad = False

    TargetTargetError = criteria(predictionOnTargetImages, targetLabelsT)

    TargetTargetError.backward()

    TargetOptimizor.step()

    targetError = TargetTargetError

    # measure batch time
    batch_time = time.time() - end
    end = time.time()

    return batch_time, data_time, DError.data.cpu().numpy()[0], targetError.data.cpu().numpy()[0]

def test(testdir, model, criteria_loss):
    print("=> Testing network... ")
    # switch to test mode
    model.eval().cuda()

    end = time.time()
    
    # load test data
    f = h5py.File(testdir + '/test_data.mat')
    data = f['data']
    labels = f['labels']

    descDistsall = []

    for i in range(data.shape[1]):
        label = torch.Tensor(f[labels[0,i]][:])
        #label = label.cuda(async=True)

        data_p1 = torch.FloatTensor(f[f['data'][0,i]]['voxelGridTDF'][:].reshape(1,1,30,30,30))
        data_p2 = torch.FloatTensor(f[f['data'][1,i]]['voxelGridTDF'][:].reshape(1,1,30,30,30))

        label = torch.autograd.Variable(label.cuda(), volatile=True)
        data_p1 = torch.autograd.Variable(data_p1.cuda(), volatile=True)
        data_p2 = torch.autograd.Variable(data_p2.cuda(), volatile=True)

        # compute output
        DesP1 = model(data_p1)
        DesP2 = model(data_p2)

        loss = criteria_loss(DesP1, DesP2, label)
        loss = loss.data.cpu().numpy()[0]

        DesP1, DesP2 = DesP1.data.cpu().numpy(), DesP2.data.cpu().numpy()
        descDists = np.sqrt(np.sum((DesP1 - DesP2)**2))

        descDistsall.append(descDists)
        # measure accuracy and record loss
        
    error_rate = ErrorRate95Recall(array(descDistsall), testdir)

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