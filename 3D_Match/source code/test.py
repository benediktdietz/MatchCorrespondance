import numpy as np
import h5py
from numpy import array
import torch
import torch.nn as nn

#gridPtsCamX,gridPtsCamY,gridPtsCamZ = np.mgrid[-0.31225162:-0.02225162:0.01,-0.0637635:0.2262365:0.01,4.28:4.57:0.01]

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
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # maxpool
            nn.MaxPool3d(kernel_size=2, stride=2),
            # conv3
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # conv6
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # conv7
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # conv8
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
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
        print(diff.size())
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        print(dist_sq.size())

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        print(torch.pow(dist, 2).size())
        print(y.size())
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

net = Match3DNet().cuda()
input = torch.autograd.Variable(torch.randn(3,1,30,30,30).cuda())
output1,output2,output3 = net(input,input,input)
label = torch.FloatTensor(np.ones(3)).cuda()
label = torch.autograd.Variable(label)
loss = ContrastiveLoss()
output = loss(output1,output2,label)