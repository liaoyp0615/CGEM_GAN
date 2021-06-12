# File to check trained model

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# import ML package
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
from typing import List, Dict, Tuple, Any
import sys
import argparse
import logging
import os
import ast
#from models.model1D import *
from datasets.dataset_1D import *


if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run histograms testing.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--datafile', action='store', type=str,
                        help='ROOT file paths')
    parser.add_argument('--G_restore_pkl_path', action='store', type=str,
                        help='Model pkl file paths')
    parser.add_argument('--outfile', action='store', type=str,
                        help='outfile file paths')
    parser.add_argument('--num_check', action='store', type=int, default=1500,
                        help='number of checked data')
    parser.add_argument('--noise_dim', action='store', type=int, default=512,
                        help='dimension of noise')
    return parser

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()

def check_fake_data(model, device, num_check, datafile):
    model.eval()
    F_Sum,F_x_Mean,F_x_Std = list(),list(),list()
    for i in range(num_check):
        noise = noiseSam(1, noise_dim)
        with torch.no_grad():
            tmp = model(noise.to(device))
            tmp = tmp.detach().cpu().numpy()
            _sum = tmp.sum(1)
            cut_x = tmp[1:91]
            weights = np.arange(0.5,3.5,3/90)
            xtmp = cut_x*weights
            if _sum!=0 :
                x_mean = xtmp.sum()/_sum
                x_std = np.sqrt(((xtmp-x_mean)**2).sum())/_sum
                F_Sum.append(_sum)
                F_x_Mean.append(x_mean)
                F_x_Std.append(x_std)
            else:
                continue
    return F_Sum, F_x_Mean, F_x_Std

def check_real_data(num_check, datafile):
    f = uproot.open(datafile)
    i=0
    R_Sum,R_x_Mean,R_x_Std = list(),list(),list()
    while(i<num_check):
        h1_name = 'h1_'+str(i)
        tmp = f[h1_name]
        if max(tmp)!=0:
            tmp = np.asarray(tmp)
            _sum = tmp.sum(1)
            cut_x = tmp[1:91]
            weights = np.arange(0.5,3.5,3/90)
            xtmp = cut_x*weights
            x_mean = xtmp.sum()/_sum
            x_std = np.sqrt(((xtmp-x_mean)**2).sum())/_sum
            R_Sum.append(_sum)
            R_x_Mean.append(x_mean)
            R_x_Std.append(x_std)
            i+=1
        else:
            i+=1
            continue
    f.close()
    return R_Sum, R_x_Mean, R_x_Std



def plot_all_info():
    fig1 = plt.figure(figsize=(12,12),dpi=100)
    ## Mean
    plt.subplot(2,2,1) #x
    plt.hist( F_x_Mean,bins=400, range=(1,3), color='steelblue', histtype='step',label="fake data")
    plt.hist( R_x_Mean,bins=400, range=(1,3), color = 'red', histtype='step', label="real data")
    plt.xlabel("x diff dist/mm")
    plt.legend()
    ## Std
    plt.subplot(2,2,2) #x
    plt.hist( F_x_Std,bins=100, range=(0,1), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_x_Std,bins=100, range=(0,1), color = 'red', histtype='step', label="real data")
    plt.xlabel("x diff Std/mm")
    plt.legend()
    ## Sum
    plt.subplot(2,2,3)
    plt.hist( F_Sum,bins=300, range=(0,40000), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_Sum,bins=300, range=(0,40000), color = 'red', histtype='step', label="real data")
    plt.xlabel("nums of e-/")
    plt.legend()
    ## histogram
    noise = noiseSam(1, noise_dim)
    tmp = G(noise.to(device))
    tmp = tmp.detach().cpu().numpy()
    _sum = tmp.sum(1)
    cut_x = tmp[1:91]
    plt.subplot(2,2,4) #x
    value = cut_x.tolist()
    axis = np.arange(0.5, 3.5, 3/90)
    plt.plot(axis, value)
    plt.xlabel("mm(x)")
    fig1.savefig(outfile)
    plt.close()
    logger.info('Save figures, done!')


if __name__=='__main__':
        logger.info("Start...")
        parser = get_parser()
        parse_args = parser.parse_args()
        datafile = parse_args.datafile
        num_check = parse_args.num_check
        G_restore_pkl_path = parse_args.G_restore_pkl_path
        outfile = parse_args.outfile
        noise_dim = parse_args.noise_dim

        # --- set up all the logging stuff
        formatter = logging.Formatter(
             '%(asctime)s - %(name)s'
             '[%(levelname)s]: %(message)s'
        )
        hander = logging.StreamHandler(sys.stdout)
        hander.setFormatter(formatter)
        logger.addHandler(hander)
        #########################################

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(device)

        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        noiseSam=get_noise_sampler()

        #G = Generator(noise_dim).to(device)
        #G.load_state_dict(torch.load(G_restore_pkl_path))
        G = torch.jit.load(G_restore_pkl_path)
        G.eval()
        logger.info('Start checking...  GAN model load state:')
        logger.info(G_restore_pkl_path)

        F_Sum, F_x_Mean, F_x_Std = check_fake_data(G, device, num_check, datafile)
        logger.info('Fake data checked!')
        R_Sum, R_x_Mean, R_x_Std = check_real_data(num_check, datafile)
        logger.info('Real data checked!')
        plot_all_info()
        logger.info('Done!')

