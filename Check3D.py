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
from models.model2D import *
from dataset import *


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
    return parser


def check_fake_data(model, device, num_check, datafile):
    model.eval()
    F_Sum,F_x_Mean,F_z_Mean,F_t_Mean,F_x_Std,F_z_Std,F_t_Std = list(),list(),list(),list(),list(),list(),list()
    for i in range(num_check):
        data = HistoDataset(datafile,1).load_data()
        with torch.no_grad():
            tmp,_,_ = model(data.to(device))
            tmp = tmp.detach().cpu().numpy()
            tmp = tmp.reshape(92,92,92)
            _t,_x,_z,_sum = tmp.sum(0).sum(0), tmp.sum(1).sum(1), tmp.sum(0).sum(1), tmp.sum(0).sum(0).sum(0)
            cut_x,cut_z,cut_t = _x[1:91],_z[1:91],_t[1:91]
            weights = np.arange(0.5,3.5,3/90)
            xtmp,ztmp,ttmp = cut_x*weights,cut_z*weights,cut_t*weights
            if _sum!=0 :
                x_mean,z_mean,t_mean = xtmp.sum()/_sum, ztmp.sum()/_sum, ttmp.sum()/_sum
                x_std,z_std,t_std = np.sqrt(((xtmp-x_mean)**2).sum())/_sum, np.sqrt(((ztmp-z_mean)**2).sum())/_sum, np.sqrt(((ttmp-t_mean)**2).sum())/_sum
                F_Sum.append(_sum)
                F_x_Mean.append(x_mean)
                F_z_Mean.append(z_mean)
                F_t_Mean.append(t_mean)
                F_x_Std.append(x_std)
                F_z_Std.append(z_std)
                F_t_Std.append(t_std)
            else:
                continue
    return F_Sum, F_x_Mean, F_z_Mean, F_t_Mean, F_x_Std, F_z_Std, F_t_Std

def check_real_data(num_check, datafile):
    f = uproot.open(datafile)
    i=0
    R_Sum,R_x_Mean,R_z_Mean,R_t_Mean,R_x_Std,R_z_Std,R_t_Std = list(),list(),list(),list(),list(),list(),list()
    while(i<num_check):
        h3_name = 'h3_'+str(i)
        tmp = f[h3_name]
        if max(tmp)!=0:
            tmp = np.asarray(tmp).reshape(92,92,92)
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



def plot_all_info():
    fig1 = plt.figure(figsize=(18,18),dpi=100)
    ## Mean
    plt.subplot(4,3,1) #x
    plt.hist( F_x_Mean,bins=400, range=(1,3), color='steelblue', histtype='step',label="fake data")
    plt.hist( R_x_Mean,bins=400, range=(1,3), color = 'red', histtype='step', label="real data")
    plt.xlabel("x diff dist/mm")
    plt.legend()
    plt.subplot(4,3,2) #z
    plt.hist( F_z_Mean,bins=400, range=(1,3), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_z_Mean,bins=400, range=(1,3), color = 'red', histtype='step', label="real data")
    plt.xlabel("z diff dist/mm")
    plt.legend()
    plt.subplot(4,3,3) #t
    plt.hist( F_t_Mean,bins=400, range=(1,3), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_t_Mean,bins=400, range=(1,3), color = 'red', histtype='step', label="real data")
    plt.xlabel("t diff dist/mm")
    plt.legend()
    ## Std
    plt.subplot(4,3,4) #x
    plt.hist( F_x_Std,bins=100, range=(0,1), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_x_Std,bins=100, range=(0,1), color = 'red', histtype='step', label="real data")
    plt.xlabel("x diff Std/mm")
    plt.legend()
    plt.subplot(4,3,5) #z
    plt.hist( F_z_Std,bins=100, range=(0,1), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_z_Std,bins=100, range=(0,1), color = 'red', histtype='step', label="real data")
    plt.xlabel("z diff Std/mm")
    plt.legend()
    plt.subplot(4,3,6) #t
    plt.hist( F_t_Std,bins=100, range=(0,1), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_t_Std,bins=100, range=(0,1), color = 'red', histtype='step', label="real data")
    plt.xlabel("t diff Std/mm")
    plt.legend()
    ## Sum
    plt.subplot(4,3,7)
    plt.hist( F_Sum,bins=300, range=(0,40000), color = 'steelblue', histtype='step', label="fake data")
    plt.hist( R_Sum,bins=300, range=(0,40000), color = 'red', histtype='step', label="real data")
    plt.xlabel("nums of e-/")
    plt.legend()
    ## histogram
    data = HistoDataset(datafile,1).load_data()
    tmp,_,_ = model(data.to(device))
    tmp = tmp.detach().cpu().numpy()
    tmp = tmp.reshape(92,92,92)
    _t,_x,_z,_sum = tmp.sum(0).sum(0), tmp.sum(1).sum(1), tmp.sum(0).sum(1), tmp.sum(0).sum(0).sum(0)
    cut_x,cut_z,cut_t = _x[1:91],_z[1:91],_t[1:91]
    plt.subplot(4,3,8) #x
    value = cut_x.tolist()
    axis = np.arange(0.5, 3.5, 3/90)
    plt.plot(axis, value)
    plt.xlabel("mm(x)")
    plt.subplot(4,3,9) #z
    value = cut_z.tolist()
    axis = np.arange(0.5, 3.5, 3/90)
    plt.plot(axis, value)
    plt.xlabel("mm(z)")
    plt.subplot(4,3,10) #t
    value = cut_t.tolist()
    axis = np.arange(0.5, 3.5, 3/90)
    plt.plot(axis, value)
    plt.xlabel("mm(t)")
    fig1.savefig(outfile)
    plt.close()
    logger.info('Save figures, done!')


if __name__=='__main__':
        logger.info("Start...")
        parser = get_parser()
        parse_args = parser.parse_args()

        datafile = parse_args.datafile
        num_check = parse_args.num_check
        model_restore_pt_path = parse_args.model_restore_pt_path
        outfile = parse_args.outfile

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

        model = CVAE().to(device)
        model.load_state_dict(torch.load(model_restore_pt_path))
        logger.info('Start checking...  VAE model load state:')
        logger.info(model_restore_pt_path)

        d_l_hist, g_l_hist, d_a1_hist, d_a2_hist, g_a_hist = list(), list(), list(), list(), list()
        d_r_loss_hist, d_f_loss_hist, d_r_acc_hist, d_f_acc_hist, g_loss_hist = list(), list(), list(), list(), list()
        F_Sum, F_x_Mean, F_z_Mean, F_t_Mean, F_x_Std, F_z_Std, F_t_Std = check_fake_data(model, device, num_check, datafile)
        logger.info('Fake data checked!')
        R_Sum, R_x_Mean, R_z_Mean, R_t_Mean, R_x_Std, R_z_Std, R_t_Std = check_real_data(num_check, datafile)
        logger.info('Real data checked!')
        plot_all_info()
        logger.info('Done!')

