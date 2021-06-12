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

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(92+3, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, 1)
        self.elu = torch.nn.ELU()

    def cal_label(self,x): #(n,1,92,92,92)  # can copy
        x = x.view(x.shape[0],92)
        _x = x.sum(1)
        cut_x = _x[:,1:91]
        _sum = cut_x.sum(-1) # (n,)
        weights = torch.arange(0.5,3.5,3/90).cuda()
        xtmp = cut_x*weights
        x_mean = xtmp.sum(-1)/_sum
        x_std = torch.sqrt(((xtmp-x_mean.view(x_mean.shape[0],1))**2).sum(-1))/_sum
        label = torch.stack((_sum,x_mean,x_std),1)
        return label

    def forward(self, x):
        label = self.cal_label(x)
        x = torch.cat((x,label),-1)
        x = self.elu(self.map1(x))
        x = F.dropout(x)
        x = self.elu(self.map2(x))
        return torch.sigmoid( self.map3(x) )

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(noise_dim, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, 92)
        self.xfer = torch.nn.SELU()
        self.xfer1 = torch.nn.CELU()
        self.xfer2 = torch.nn.Softplus()
        self.xfer3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.xfer( self.map1(x) )
        x = F.dropout(x)
        x = self.xfer( self.map2(x) )
        x = F.dropout(x)
        x = self.xfer3( self.map3( x ) )
        #x = abs(x)
        return x

