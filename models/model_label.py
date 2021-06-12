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
        self.map1 = nn.Linear(1, hidden_size) # only learn the polya distribution
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, 1)
        self.elu = torch.nn.ELU()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        x = self.elu(self.map3(x))
        return torch.sigmoid( self.map4(x) )

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(noise_dim, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, 1)
        self.xfer = torch.nn.SELU()
        self.xfer1 = torch.nn.CELU()
        self.xfer2 = torch.nn.Softplus()
        self.xfer3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.xfer1( self.map1(x) )
        x = self.xfer1( self.map2(x) )
        x = self.xfer1( self.map3(x) )
        x = self.xfer1( self.map4(x) ) # only learn the polya distribution
        return x

