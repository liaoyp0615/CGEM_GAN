# File to load histogram data

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
import random
import uproot

class HistoDataset(Dataset):
    #load input data.
    def __init__(self, datafile, num):
        self.datafile = datafile
        self.num = num
        f = uproot.open(self.datafile)
        histograms = []
        i=0
        while(i<self.num):
            k = random.randint( 0, len(f.keys())-1 ) # randint includes min and max.
            h1_name = 'h1_'+str(k)
            tmp = f[h1_name]
            if max(tmp)!=0:
                tmp = np.asarray(tmp).reshape(1,92)
                histograms.append(tmp)
                i+=1
            else:
                continue
        f.close()
        self.histograms = histograms
 
    def __getitem__(self, index):
        hist = self.histograms[index]
        hist = np.asarray(hist).astype(int)
        hist = torch.Tensor(hist)
        return hist
    
    def load_data(self):
        data = np.asarray(self.histograms).reshape(self.num,1,92)
        data = torch.Tensor(data)
        return data

    def __len__(self):
        return len(self.histograms)
