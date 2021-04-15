# import ML package
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Tuple, Any


class Discriminator_9L(nn.Module):
    def __init__(self):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv3d(1,12,4,1,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(12,24,4,1,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(24,36,4,2,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(36,48,4,1,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(48,60,5,2,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(60,72,4,1,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(72,84,5,2,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(84,96,4,1,0),
                    nn.LeakyReLU(),
                    nn.Conv3d(96,128,3,1,0),
                    nn.LeakyReLU(),
                    )
        self.out = nn.Linear(128,1)
        T_init = torch.randn(128*1*1*1, 64*16) # B=128, C=16
        self.T = nn.Parameter(T_init, requires_grad=True)
        self.out_minidisc = nn.Linear(128*1*1*1 + 64, 1)


    def cal_label(self,x): #(n,1,92,92,92)  # can copy
        x = x.reshape(x.shape(0),92,92,92)
        _t = x.sum(1).sum(1) # t direction distribution
        _x = x.sum(-1).sum(-1) # x direction distribution
        _z = x.sum(1).sum(-1) # z direction distribution  (n,92)
        print('_x:',_x)
        cut_x,cut_z,cut_t = _x(:,1:91), _z(:,1:91), _t(:,1:91)
        print(cut_x.shape)
        _sum = cut_x.sum(-1) # (n,)
        print('_sum:',_sum)
        weights = np.arange(0.5,3.5,3/90).reshape(x.shape(0),90)
        print('weights:',weights)
        xtmp,ztmp,ttmp = cut_x*weights, cut_z*weights, cut_t*weights
        print('xtmp.shape:',xtmp.shape)
        x_mean,z_mean,t_mean = xtmp.sum(-1)/_sum, ztmp.sum(-1)/_sum, ttmp.sum(-1)/_sum #(n,)
        print('x_mean.shape',x_mean.shape)
        print('x_mean',x_mean)
        x_std = /_sum
        x_std,z_std,t_std = np.sqrt(((xtmp-x_mean)**2).sum(-1))/_sum, np.sqrt(((ztmp-z_mean)**2).sum(-1))/_sum, np.sqrt(((ttmp-t_mean)**2).sum(-1))/_sum
        lable = torch.cat((_sum,x_mean,z_mean,t_mean,x_std,z_std,t_std),-1)
        return label


    def forward(self, x, matching=False, minidisc=False, wgan=False):
        label = cal_label(x)
        x = self.conv(x)
        feature = x.view(x.size(0),-1) # (n, 128*1*1*1)
        x = torch.cat((feature,label),-1)

        if minidisc:
            T = self.T
            T = T.to(device)
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
        else:
            x = self.out(x) #1

        if wgan:
            output  = x
        else:
            output  = torch.sigmoid(x)

        if matching == True:
            return feature,output
        else:
            return output

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = 2 * F.sigmoid(x)
        return x

class Generator_9L(nn.Module):
    def __init__(self, noise_dim):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(noise_dim, 512),
                    nn.ReLU(),
                    #nn.BatchNorm1d(1024),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    #nn.BatchNorm1d(4*4*4*24)
                    )
        self.inconv = nn.Sequential(
                    nn.ConvTranspose3d(128,96,3,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(96,84,4,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(84,72,5,2,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(72,60,4,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(60,48,5,2,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(48,36,4,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(36,24,4,2,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(24,12,4,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(12,1,4,1,0),
                    Act_op()
                    )
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128,1,1,1)
        x = self.inconv(x)
        return x


class get_noise_sampler():
    def forward(self):
        return lambda m, n: torch.rand(m, n).requires_grad_()


class Discriminator_4L(nn.Module):
    def __init__(self):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv3d(1,16,6,2,0), # 92->44
                    nn.LeakyReLU(),
                    nn.Conv3d(16,32,5,3,0), #44->20
                    nn.LeakyReLU(),
                    nn.Conv3d(32,48,5,3,0), #20->8
                    nn.LeakyReLU(),
                    nn.Conv3d(48,64,3,1,0), #8->2
                    nn.LeakyReLU(),
                    )
        self.out = nn.Linear(64*2*2*2,1)
        T_init = torch.randn(64*2*2*2, 64*16) # B=128, C=16
        self.T = nn.Parameter(T_init, requires_grad=True)
        self.out_minidisc = nn.Linear(64*2*2*2 + 64, 1)

    def forward(self, x, matching=False, minidisc=False, wgan=False):
        x = self.conv(x)
        feature = x.view(x.size(0),-1)
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
        else:
            x = self.out(x) #1

        if wgan:
            output  = x
        else:
            output  = torch.sigmoid(x)

        if matching == True:
            return feature,output
        else:
            return output

class Generator(nn.Module):
    def __init__(self, noise_dim):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(noise_dim, 2048),
                    nn.ReLU(),
                    #nn.BatchNorm1d(1024),
                    nn.Linear(2048, 2*2*2*128),
                    nn.ReLU(),
                    #nn.BatchNorm1d(4*4*4*24)
                    )
        self.inconv = nn.Sequential(
                    nn.ConvTranspose3d(128,64,3,1,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(64,32,5,3,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(32,16,5,3,0),
                    nn.ReLU(),
                    nn.ConvTranspose3d(16,1,6,2,0),
                    Act_op()
                    )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128,2,2,2)
        x = self.inconv(x)
        return x
