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
                    nn.Conv2d(16,32,5,3,0), #44->20
                    nn.LeakyReLU(),
                    nn.Conv2d(32,48,5,3,0), #20->8
                    nn.LeakyReLU(),
                    nn.Conv2d(48,64,3,1,0), #8->2
                    nn.LeakyReLU(),
                    )
        self.out = nn.Linear(64*2*2+5,1)
        T_init = torch.randn(64*2*2, 64*16) # B=128, C=16
        self.T = nn.Parameter(T_init, requires_grad=True)
        self.out_minidisc = nn.Linear(64*2*2 + 64, 1)

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

        if minidisc:
            T = self.T
            T = T.cuda()
            M = feature.mm(T)
            M = M.view(-1, 64,16)  #(B,C)
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
            out_T = torch.cat(tuple(out_tensor)).view(M.size()[0],64) #B
            x = torch.cat((feature,out_T),1)  # (n, 128*1*1*1+64)
            x = self.out_minidisc(x) #1
            output = torch.sigmoid(x)
        else:
            x = self.out(x)
            output = torch.sigmoid(x)

        if matching == True:
            return feature,output
        else:
            return output

class Generator_4L(nn.Module):
    def __init__(self, noise_dim):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Generator_4L, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(noise_dim, 2048),
                    nn.ReLU(),
                    #nn.BatchNorm1d(1024),
                    nn.Linear(2048, 2*2*128),
                    nn.ReLU(),
                    #nn.BatchNorm1d(4*4*4*24)
                    )
        self.inconv = nn.Sequential(
                    nn.ConvTranspose2d(128,64,3,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64,32,5,3,0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32,16,5,3,0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16,1,6,2,0),
                    Act_op()
                    )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128,2,2)
        x = self.inconv(x)
        return x
