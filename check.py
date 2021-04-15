import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Tuple, Any
from sklearn.utils import shuffle
import random
import uproot
from model import *

class CheckData(nn.Module):
    def __

    def load_check_real_data(self):
        f = uproot.open(self.datafile)
        i=0
        R_Sum,R_x_Mean,R_z_Mean,R_t_Mean,R_x_Std,R_z_Std,R_t_Std = list(),list(),list(),list(),list(),list(),list()
        while(i<self.num):
            h3_name = 'h3_'+str(i)
            tmp = f[h3_name]
            if max(tmp)!=0:
                tmp = np.asarray(tmp)
                tmp = tmp.reshape(92,92,92)
                _t,_x,_z,_sum = tmp.sum(0).sum(0), tmp.sum(1).sum(1), tmp.sum(0).sum(1), tmp.sum(0).sum(0).sum(0)
                cut_x,cut_z,cut_t = _x[1:91],_z[1:91],_t[1:91]
                weights = np.arange(0.5,3.5,3/90)
                xtmp,ztmp,ttmp = cut_x*weights,cut_z*weights,cut_t*weights
                x_mean,z_mean,t_mean = xtmp.sum()/_sum, ztmp.sum()/_sum, ttmp.sum()/_sum
                x_std,z_std,t_std = np.sqrt(((xtmp-x_mean)**2).sum())/_sum, np.sqrt(((ztmp-z_mean)**2).sum())/_sum, np.sqrt(((ttmp-t_mean)**2).sum())/_sum
                R_Sum.append(_sum)
                R_x_Mean.append(x_mean)
                R_z_Mean.append(z_mean)
                R_t_Mean.append(t_mean)
                R_x_Std.append(x_std)
                R_z_Std.append(z_std)
                R_t_Std.append(t_std)
                i+=1
            else:
                i+=1
                continue
        f.close()
        return R_Sum, R_x_Mean, R_z_Mean, R_t_Mean, R_x_Std, R_z_Std, R_t_Std

    def check_fake_data(num):
        F_Sum,F_x_Mean,F_z_Mean,F_t_Mean,F_x_Std,F_z_Std,F_t_Std = list(),list(),list(),list(),list(),list(),list()
        
