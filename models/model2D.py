# import ML package
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Tuple, Any


class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = 2 * torch.sigmoid(x)
        return x


class Discriminator_4L(nn.Module):
    def __init__(self):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Discriminator_4L, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(1,16,6,2,0), # 92->44
                    nn.LeakyReLU(),
                    #nn.BatchNorm2d(16),
                    nn.Conv2d(16,32,5,3,0), #44->14
                    nn.LeakyReLU(),
                    #nn.BatchNorm2d(32),
                    nn.Conv2d(32,64,5,3,0), #14->4
                    nn.LeakyReLU(),
                    #nn.BatchNorm2d(64),
                    nn.Conv2d(64,128,4,1,0), #4->1
                    nn.LeakyReLU(),
                    )
        self.out = nn.Sequential(
                   nn.Linear(128*1*1+5, 3000),
                   nn.LeakyReLU(),
                   nn.Linear(3000, 300),
                   nn.LeakyReLU(),
                   nn.Linear(300, 1),
                   )
        T_init = torch.randn(128*1*1+5, 48*16) # B=128, C=16
        self.T = nn.Parameter(T_init, requires_grad=True)
        self.out_minidisc = nn.Sequential(
                   nn.Linear(128*1*1+5+48, 3000),
                   nn.LeakyReLU(),
                   nn.Linear(3000, 300),
                   nn.LeakyReLU(),
                   nn.Linear(300, 1),
                   )

    def cal_label(self,x): #(n,1,92,92,92)  # can copy
        x = x.view(x.shape[0],92,92)
        _x,_z = x.sum(2),x.sum(1)
        cut_x,cut_z = _x[:,1:91], _z[:,1:91]
        _sum = cut_x.sum(-1) # (n,)
        weights = torch.arange(0.5,3.5,3/90).cuda()
        xtmp,ztmp = cut_x*weights, cut_z*weights
        x_mean,z_mean = xtmp.sum(-1)/_sum, ztmp.sum(-1)/_sum #(n,)
        x_std,z_std = torch.sqrt(((xtmp-x_mean.view(x_mean.shape[0],1))**2).sum(-1))/_sum, torch.sqrt(((ztmp-z_mean.view(z_mean.shape[0],1))**2).sum(-1))/_sum
        label = torch.stack((_sum,x_mean,z_mean,x_std,z_std),1)
        return label

    def forward(self, x, matching=False, minidisc=False):
        label = self.cal_label(x) # 5
        x = self.conv(x)
        feature = x.view(x.size(0),-1)
        feature = torch.cat((feature,label),-1)
        #print("feature shape:",feature.shape)
        x = feature
        #print(x) #debug

        '''
        T = self.T
        T = T.cuda()
        M = feature.mm(T)
        M = M.view(-1, 48,16)  #(B,C)
        out_tensor = []
        for i in range(M.size()[0]):
            out_i = None
            for j in range(M.size()[0]):
                o_i = torch.sum(torch.abs(M[i,...]-M[j,...]),1)
                o_i = torch.exp(o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i
            out_tensor.append(out_i)
        out_T = torch.cat(tuple(out_tensor)).view(M.size()[0],48) #B
        x = torch.cat((feature,out_T),1)  # (n, 128*1*1*1+64)
        x = self.out_minidisc(x) #1
        '''
        x = self.out(x)
        #output = torch.sigmoid(x)
        output = x
        return output

class Generator_4L(nn.Module):
    def __init__(self, noise_dim):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Generator_4L, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(noise_dim,2000),
                    nn.ReLU(),
                    nn.BatchNorm1d(2000),
                    nn.Linear(2000,1000),
                    nn.ReLU(),
                    nn.BatchNorm1d(1000),
                    nn.Linear(1000, 2*2*256),
                    nn.ReLU(),
                    nn.BatchNorm1d(2*2*256)
                    )
        self.inconv = nn.Sequential(
                    nn.ConvTranspose2d(256,128,3,1,0), #4
                    nn.ReLU(),
                    nn.ConvTranspose2d(128,64,3,2,0), #9
                    nn.ReLU(),
                    nn.ConvTranspose2d(64,32,5,2,0), #21
                    nn.ReLU(),
                    nn.ConvTranspose2d(32,16,5,2,0), #45
                    nn.ReLU(),
                    nn.ConvTranspose2d(16,1,4,2,0), #92
                    #Act_op(),
                    nn.ReLU(),
                    )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 256,2,2)
        x = self.inconv(x)
        return x
