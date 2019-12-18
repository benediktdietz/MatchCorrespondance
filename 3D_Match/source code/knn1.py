import numpy as np
import time
import torch

n1 = torch.rand(20000,3).cuda()
n2 = torch.rand(21000,3).cuda()

end = time.time()

sum_1 = torch.t(torch.sum(n1**2, 1).repeat(n2.size()[0],1))

sum_2 = torch.sum(n2**2, 1).repeat(n1.size()[0],1)

knnDist,_ = torch.min(torch.addmm(1.0,sum_1+sum_2,-2.0,n1,torch.t(n2)),0)
knnDist = torch.sqrt(knnDist)

print(time.time() - end)