# import ROOT 
from ROOT import gROOT
from ROOT import TFile as tf

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
import torch.backends.cudnn as cudnn
from typing import List, Dict, Tuple, Any

# Using uproot for loading, which is less time-consuming.
from sklearn.utils import shuffle
import random
import uproot

import sys
import argparse
import logging
import os
import ast

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
        description='Run histograms training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--datafile', action='store', type=str,
                        help='ROOT file paths')
    parser.add_argument('--outfile', action='store', type=str,
                        help='outfile file paths')
    parser.add_argument('--num_epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--d_minibatch_size', action='store', type=int, default=40,
                        help='d_minibatch_size')
    parser.add_argument('--g_minibatch_size', action='store', type=int, default=40,
                        help='g_minibatch_size')
    parser.add_argument('--noise_dim', action='store', type=int, default=400,
                        help='noise_input_dimention')
    parser.add_argument('--d_lr', action='store', type=float, default=2e-4,
                        help='discriminator learning rate')
    parser.add_argument('--g_lr', action='store', type=float, default=2e-4,
                        help='generator learning rate')
    parser.add_argument('--D_restore_pkl_path', action='store', type=str,
                        help='D restore_pkl_path pkl file paths')
    parser.add_argument('--D_pkl_path', action='store', type=str,
                        help='D pkl_path pkl file paths')
    parser.add_argument('--G_restore_pkl_path', action='store', type=str,
                        help='G restore_pkl_path pkl file paths')
    parser.add_argument('--G_pkl_path', action='store', type=str,
                        help='G pkl_path pkl file paths')
    parser.add_argument('--restore', action='store', type=ast.literal_eval, default=False,
                        help='ckpt file paths')
    parser.add_argument('--wgan', action='store', type=ast.literal_eval, default=True,
                        help='w-GAN')
    parser.add_argument('--gradient_penalty', action='store', type=ast.literal_eval, default=False,
                        help='gp')
    parser.add_argument('--feature_matching', action='store', type=ast.literal_eval, default=False,
                        help='fm')
    parser.add_argument('--minidisc', action='store', type=ast.literal_eval, default=False,
                        help='md')
    return parser
    
def load_data(datafile, num):
    "load num random data."
    f = uproot.open(datafile)
    hist = []
    i=0
    while(i<num):
        k = random.randint( 0, len(f.keys())-1 ) # randint includes min and max.
        h2_name = 'h2_'+str(k)
        tmp = f[h2_name]
        if max(tmp)!=0:
            hist.append(tmp)
            i+=1
        else:
            continue
    hist = np.asarray(hist)
    hist = hist.reshape(hist.shape[0],92,92)
    f.close()
    return hist

def load_check_real_data(datafile, num):
    "load data as series."
    f = uproot.open(datafile)
    i=0
    R_Sum = []
    R_x_Mean = []
    R_z_Mean = []
    R_x_Std = []
    R_z_Std = []
    while(i<num):
        h2_name = 'h2_'+str(i)
        tmp = f[h2_name]
        if max(tmp)!=0:
            tmp = np.asarray(tmp)
            tmp = tmp.reshape(92,92)
            _x = tmp.sum(axis=1) # x direction distribution
            _z = tmp.sum(axis=0) # z direction distribution
            cut_x = _x[1:91]
            cut_z = _z[1:91]
            _sum = cut_x.sum()

            weights = np.arange(0.5,3.5,3/90)
            xtmp = cut_x*weights
            ztmp = cut_z*weights
            x_mean = xtmp.sum()/_sum
            z_mean = ztmp.sum()/_sum

            x_std = np.sqrt(((xtmp-x_mean)**2).sum())/_sum
            z_std = np.sqrt(((ztmp-z_mean)**2).sum())/_sum

            R_Sum.append(_sum)
            R_x_Mean.append(x_mean)
            R_z_Mean.append(z_mean)
            R_x_Std.append(x_std)
            R_z_Std.append(z_std)
            i+=1
        else:
            i+=1
            continue
    f.close()
    return R_Sum, R_x_Mean, R_z_Mean, R_x_Std, R_z_Std

def check_fake_data(num):
    nevt=num
    # --- Fake data check
    F_Sum = []
    F_x_Mean = []
    F_z_Mean = []
    F_x_Std = []
    F_z_Std = []
    for i in range(nevt):
        noise = noiseSam(1, noise_dim)
        fake_data = G(noise)
        fake_hist = fake_data.detach().to(Device).numpy()
        tmp = fake_hist.reshape(92,92)
        _x = tmp.sum(axis=1) # x direction distribution
        _z = tmp.sum(axis=0) # z direction distribution
        cut_x = _x[1:91]
        cut_z = _z[1:91]
        _sum = cut_x.sum()

        weights = np.arange(0.5,3.5,3/90)
        xtmp = cut_x*weights
        ztmp = cut_z*weights
        if _sum!=0 :
            x_mean = xtmp.sum()/_sum
            z_mean = ztmp.sum()/_sum

            x_std = np.sqrt(((xtmp-x_mean)**2).sum())/_sum
            z_std = np.sqrt(((ztmp-z_mean)**2).sum())/_sum

            F_Sum.append(_sum)
            F_x_Mean.append(x_mean)
            F_z_Mean.append(z_mean)
            F_x_Std.append(x_std)
            F_z_Std.append(z_std)
        else:
            continue
    return F_Sum, F_x_Mean, F_z_Mean, F_x_Std, F_z_Std


def plot_all_info():
    fig1 = plt.figure(figsize=(18,18),dpi=100)
    
    ## Mean
    #x
    plt.subplot(3,3,1)
    plt.hist( F_x_Mean,bins=400, range=(1,3),
         color = 'steelblue',
         histtype='step',
         label="fake data")
    plt.hist( R_x_Mean,bins=400, range=(1,3),
         color = 'red',
         histtype='step',
         label="real data")
    plt.xlabel("x diff dist/mm")
    plt.legend()
    #z
    plt.subplot(3,3,2)
    plt.hist( F_z_Mean,bins=400, range=(1,3),
         color = 'steelblue',
         histtype='step',
         label="fake data")
    plt.hist( R_z_Mean,bins=400, range=(1,3),
         color = 'red',
         histtype='step',
         label="real data")
    plt.xlabel("z diff dist/mm")
    plt.legend()
    ## Std
    #x
    plt.subplot(3,3,3)
    plt.hist( F_x_Std,bins=300, range=(0,4),
         color = 'steelblue',
         histtype='step',
         label="fake data")
    plt.hist( R_x_Std,bins=300, range=(0,4),
         color = 'red',
         histtype='step',
         label="real data")
    plt.xlabel("x diff Std/mm")
    plt.legend()
    #z
    plt.subplot(3,3,4)
    plt.hist( F_z_Std,bins=300, range=(0,4),
         color = 'steelblue',
         histtype='step',
         label="fake data")
    plt.hist( R_z_Std,bins=300, range=(0,4),
         color = 'red',
         histtype='step',
         label="real data")
    plt.xlabel("z diff Std/mm")
    plt.legend()
    ## Sum
    plt.subplot(3,3,5)
    plt.hist( F_Sum,bins=300, range=(0,40000),
         color = 'steelblue',
         histtype='step',
         label="fake data")
    plt.hist( R_Sum,bins=300, range=(0,40000),
         color = 'red',
         histtype='step',
         label="real data")
    plt.xlabel("nums of e-/")
    plt.legend()

    ## histogram
    noise = noiseSam(1, noise_dim)
    fake_data = G(noise)
    fake_hist = fake_data.detach().to(Device).numpy()
    tmp = fake_hist.reshape(92,92)
    _x = tmp.sum(axis=1) # x direction distribution
    _z = tmp.sum(axis=0) # z direction distribution
    cut_x = _x[1:91]
    cut_z = _z[1:91]

    #x
    plt.subplot(3,3,6)
    value = cut_x.tolist()
    axis = np.arange(0.5, 3.5, 3/90)
    plt.plot(axis, value)
    plt.xlabel("mm(x)")

    #z
    plt.subplot(3,3,7)
    value = cut_z.tolist()
    axis = np.arange(0.5, 3.5, 3/90)
    plt.plot(axis, value)
    plt.xlabel("mm(z)")

    ###### plot history ######
    plt.subplot(3,3,8) # loss
    if wgan:
        plt.plot(d_l_hist, label='d-loss(wgan)')
        plt.plot(g_l_hist, label='g-loss(wgan)')
    else:
        plt.plot(d_r_loss_hist, label='d-real')
        plt.plot(d_f_loss_hist, label='d-fake')
        plt.plot(g_loss_hist, label='g-loss')
    plt.legend()

    plt.subplot(3,3,9) #accuracy
    if wgan:
        plt.plot(d_a1_hist, label='d-real-score')
        plt.plot(d_a2_hist, label='d-fake-score')
        plt.plot(g_a_hist,  label='gen-score')
    else:
        plt.plot(d_r_acc_hist, label='d-r-acc')
        plt.plot(d_f_acc_hist, label='d-f-acc')
    plt.legend()

    fig1.savefig(outfile)
    plt.close()
    logger.info('Save figures, done!')



def get_Projection(filename, hist):
    _f = filename
    _h2 = _f[hist-1,...]
    tmp = _h2.reshape(92,92)
    #print(kk)
    _x = tmp.sum(axis=1) # x direction distribution
    _z = tmp.sum(axis=0) # z direction distribution
    #print("x=",x,'\n', "z=",z,'\n', "t=",t,'\n')
    return _x,_z

class Discriminator(nn.Module):
    def __init__(self):
        "Out = (In + 2*Padding - k)/stride + 1"
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(1,12,4,1,0),
                    nn.LeakyReLU()
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(12,24,4,1,0),
                    nn.LeakyReLU(),
                    )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(24,36,4,2,0),
                    nn.LeakyReLU()
                    )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(36,48,4,1,0),
                    nn.LeakyReLU()
                    )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(48,60,5,2,0),
                    nn.LeakyReLU()
                    )
        self.conv6 = nn.Sequential(
                    nn.Conv2d(60,72,4,1,0),
                    nn.LeakyReLU()
                    )
        self.conv7 = nn.Sequential(
                    nn.Conv2d(72,84,5,2,0),
                    nn.LeakyReLU()
                    )
        self.conv8 = nn.Sequential(
                    nn.Conv2d(84,96,4,1,0),
                    nn.LeakyReLU()
                    )
        self.conv9 = nn.Sequential(
                    nn.Conv2d(96,128,3,1,0),
                    nn.LeakyReLU()
                    )
        self.out = nn.Linear(128,1)
        T_init = torch.randn(128*1*1, 64*16) # B=128, C=16
        self.T = nn.Parameter(T_init, requires_grad=True)
        self.out_minidisc = nn.Linear(128*1*1 + 64, 1)

    def forward(self, x, matching=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        feature = x.view(x.size(0),-1) # (n, 64*2*2*2)
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
            x = torch.cat((feature,out_T),1)
            x = self.out_minidisc(x)
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
                    nn.Linear(2048, 128),
                    nn.ReLU(),
                    #nn.BatchNorm1d(4*4*4*24)
                    )
        self.inconv1 = nn.Sequential(
                    nn.ConvTranspose2d(128,96,3,1,0),
                    nn.ReLU(),
                    )
        self.inconv2 = nn.Sequential(
                    nn.ConvTranspose2d(96,84,4,1,0),
                    nn.ReLU(),
                    )
        self.inconv3 = nn.Sequential(
                    nn.ConvTranspose2d(84,72,5,2,0),
                    nn.ReLU(),
                    )
        self.inconv4 = nn.Sequential(
                    nn.ConvTranspose2d(72,60,4,1,0),
                    nn.ReLU(),
                    )
        self.inconv5 = nn.Sequential(
                    nn.ConvTranspose2d(60,48,5,2,0),
                    nn.ReLU(),
                    )
        self.inconv6 = nn.Sequential(
                    nn.ConvTranspose2d(48,36,4,1,0),
                    nn.ReLU(),
                    )
        self.inconv7 = nn.Sequential(
                    nn.ConvTranspose2d(36,24,4,2,0),
                    nn.ReLU(),
                    )
        self.inconv8 = nn.Sequential(
                    nn.ConvTranspose2d(24,12,4,1,0),
                    nn.ReLU(),
                    )
        self.inconv9 = nn.Sequential(
                    nn.ConvTranspose2d(12,1,4,1,0),
                    nn.ReLU(),
                    )
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128,1,1)
        x = self.inconv1(x)
        x = self.inconv2(x)
        x = self.inconv3(x)
        x = self.inconv4(x)
        x = self.inconv5(x)
        x = self.inconv6(x)
        x = self.inconv7(x)
        x = self.inconv8(x)
        x = self.inconv9(x)
        return x
    
def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()

    
def train_D(): # w-gan
    d_r_score = D( real_data )
    
    noise = noiseSam(d_minibatch_size, noise_dim )
    noise = noise.cuda()
    fake_data = G( noise )
    d_f_score = D( fake_data )
    
    if gradient_penalty:
        # gradient panelty
        alpha = torch.rand((d_minibatch_size,1,1,1))
        alpha = alpha.cuda()
        hat_data = alpha*real_data + (1-alpha)*fake_data #sampling zone
        hat_data.cuda().requires_grad_()
        d_hat_score = D(hat_data)

        gradients = torch.autograd.grad(outputs=d_hat_score, inputs=hat_data,
                    grad_outputs=torch.ones(d_hat_score.size()).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        lambda_ = 10
        gp = lambda_*((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        d_loss = -torch.mean(d_r_score) + torch.mean(d_f_score) + gp
    else:
        d_loss = -torch.mean(d_r_score) + torch.mean(d_f_score)

    d_loss.backward()

    d_r_acc = torch.mean((d_r_score),0).detach().to(Device).numpy()
    d_f_acc = torch.mean((d_f_score),0).detach().to(Device).numpy()

    return d_loss, d_r_acc, d_f_acc

def train_G():
    noise = noiseSam(g_minibatch_size, noise_dim)
    noise = noise.cuda()
    fake_data = G( noise )
    g_score = D( fake_data )
    g_loss = -torch.mean(g_score)

    if feature_matching:
        r_feature, _       = D( real_data,matching=True )
        f_feature, g_score = D( fake_data,matching=True )
        r_feature = torch.mean(r_feature,0)
        f_feature = torch.mean(f_feature,0)
        criterionG = nn.MSELoss()
        g_loss_FM = criterionG(f_feature,r_feature)
        g_loss = g_loss + g_loss_FM
    g_loss.backward()
    
    return g_loss.item(), torch.mean(g_score,0).detach().to(Device).numpy()

def train_D2(): # GAN
    d_r_score = D( real_data )
    d_r_standard = torch.ones(d_minibatch_size,1)
    d_r_loss = criterion( d_r_score, d_r_standard.cuda())  # ones = true
    d_r_loss.backward()

    noise = noiseSam(d_minibatch_size, noise_dim )
    noise = noise.cuda()
    fake_data = G( noise )
    d_f_score = D( fake_data )
    #d_f_standard = torch.ones(d_minibatch_size,1)*0.01
    d_f_standard = torch.zeros(d_minibatch_size,1)
    d_f_loss = criterion( d_f_score, d_f_standard.cuda())  # zeros = fake
    d_f_loss.backward()

    d_r_acc = torch.mean((d_r_score),0).detach().to(Device).numpy()
    d_f_acc = torch.mean((d_f_score),0).detach().to(Device).numpy()
    return d_r_loss, d_f_loss, d_r_acc, d_f_acc

def train_G2():
    noise = noiseSam(g_minibatch_size, noise_dim)
    noise = noise.cuda()
    fake_data = G( noise )
    g_score = D( fake_data )
    g_standard = torch.ones(g_minibatch_size,1)
    g_loss = criterion( g_score, g_standard.cuda() )  # we want to fool, so pretend it's all genuine
    g_loss.backward()
    return g_loss

if __name__ == '__main__':

    print('start...')
    print(torch.cuda.is_available())
    #CUDA_DEVICES = 0,1,2
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    devices = []
    for i in range( torch.cuda.device_count() ):
        tmp_device = torch.device("cuda:%d"%i)
        devices.append(tmp_device)
    Device = torch.device("cpu")
    #####################################
    parser = get_parser()
    parse_args = parser.parse_args()
    num_epochs = parse_args.num_epochs
    D_restore_pkl_path = parse_args.D_restore_pkl_path
    G_restore_pkl_path = parse_args.G_restore_pkl_path
    G_pkl_path = parse_args.G_pkl_path
    D_pkl_path = parse_args.D_pkl_path
    datafile = parse_args.datafile
    outfile = parse_args.outfile
    wgan = parse_args.wgan
    restore = parse_args.restore
    gradient_penalty = parse_args.gradient_penalty
    feature_matching = parse_args.feature_matching
    minidisc = parse_args.minidisc
    d_learning_rate = parse_args.d_lr
    g_learning_rate = parse_args.g_lr
    d_minibatch_size = parse_args.d_minibatch_size
    g_minibatch_size = parse_args.g_minibatch_size
    noise_dim = parse_args.noise_dim
    #####################################
    # --- set up all the logging stuff
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )
    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)
    #####################################
    logger.info('constructing graph')

    # --- data parameters-----------------------------------------------------------
    # --- Training parameters
    epochs = num_epochs
    print_interval = num_epochs/10

    logger.info(devices)
    
    #-----------------------------------------------------------------------------------
    # --- Construct NN
    D = Discriminator().cuda()
    G = Generator(noise_dim).cuda()
    if torch.cuda.device_count()>1:
        D = nn.DataParallel(D)
        G = nn.DataParallel(G)
        cudnn.benchmark = True
    noiseSam=get_noise_sampler()
    criterion = nn.BCELoss()
    if restore:
        G.load_state_dict(torch.load(G_restore_pkl_path))
        D.load_state_dict(torch.load(D_restore_pkl_path))
        print('restored from ',G_restore_pkl_path)
        print('restored from ',D_restore_pkl_path)

    # --- Paraments of NNs --------------------------------------------------------------
    logger.info("wgan?")
    logger.info(wgan)
    logger.info("gradient penalty?")
    logger.info(gradient_penalty)
    logger.info("feature matching?")
    logger.info(feature_matching)
    logger.info("minibatch discrimination?")
    logger.info(minidisc)
    if wgan:
        g_optimizer = optim.RMSprop(G.parameters(), lr=g_learning_rate)
        d_optimizer = optim.RMSprop(D.parameters(), lr=d_learning_rate)
    else:
        d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

    d_l_hist, g_l_hist, d_a1_hist, d_a2_hist, g_a_hist = list(), list(), list(), list(), list()
    d_r_loss_hist, d_f_loss_hist, d_r_acc_hist, d_f_acc_hist, g_loss_hist = list(), list(), list(), list(), list()

    # --- Training
    for epoch in range(epochs):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # --- Set Training strategy
        if wgan:
            if epoch < 50:
                d_num=5
                g_num=1
            elif epoch>=50 and epoch <100:
                d_num=2
                g_num=1
            elif epoch>=100 and epoch <1100:
                d_num=2
                g_num=2
            else:
                d_num=1
                g_num=1
        else:
            d_num=1
            g_num=1
        
        for i in range(d_num):
            raw_data = load_data( datafile, d_minibatch_size )
            real_data = raw_data.reshape(d_minibatch_size, 1, 92, 92)
            #noise = np.random.normal(loc=0,scale=0.1,size=(d_minibatch_size, 1, 92, 92))
            #real_data = real_data + noise
            #real_data[real_data<0]=0
            '''
            pp = d_minibatch_size//10
            raw_data = load_data( datafile, d_minibatch_size-pp )
            real_data = raw_data.reshape(raw_data.shape[0], 1, 92, 92, 92)
            noise_data = np.random.normal(loc=0,scale=0.1,size=(pp, 1, 92, 92, 92))
            combined_data = np.concatenate((real_data,noise_data),0)
            noise = np.random.normal(loc=0,scale=0.1,size=(d_minibatch_size, 1, 92, 92, 92))
            #noise = np.random.choice((-2,2,0), size=(d_minibatch_size,1,92,92,92), p=[0.05,0.05,0.9])
            real_data = combined_data + noise
            real_data[real_data<0]=0
            '''

            real_data_t = torch.Tensor(real_data)
            real_data = real_data_t.cuda().requires_grad_()

            D.zero_grad() # the same
            if wgan:
                d_loss, d_r_acc, d_f_acc = train_D()
                if i==d_num-1:
                    d_l_hist.append(d_loss)
                    d_a1_hist.append(d_r_acc)
                    d_a2_hist.append(d_f_acc)
            else:
                d_r_loss, d_f_loss, d_r_acc, d_f_acc = train_D2()
                d_loss = d_r_loss + d_f_loss
                if i==d_num-1:
                    d_r_loss_hist.append(d_r_loss)
                    d_f_loss_hist.append(d_f_loss)
                    d_r_acc_hist.append(d_r_acc)
                    d_f_acc_hist.append(d_f_acc)
            d_optimizer.step()

        for i in range(g_num):
            G.zero_grad()
            if wgan:
                g_loss, g_score = train_G()
                if i==g_num-1:
                    g_l_hist.append(g_loss)
                    g_a_hist.append(g_score)
            else:
                g_loss = train_G2()
                if i==g_num-1:
                    g_loss_hist.append(g_loss)
            g_optimizer.step()   
 
        if epoch == 1:
            logger.info("1 epoch completed! This code is running successfully!")

        if( epoch % print_interval) == (print_interval-1) :
            logger.info( "Epoch %6d. G_Loss %5.3f. D_Loss %5.3f" % ( epoch+1, g_loss ,d_loss ) )
            #print( "Epoch %6d. G_Loss %5.3f. D_Loss %5.3f" % ( epoch+1, g_loss ,d_loss ) )
        
        if epoch%100==0 and epoch!=0:
            torch.save(G.state_dict(), G_pkl_path)
            torch.save(D.state_dict(), D_pkl_path)
            logger.info('Save Net state!')

    logger.info( "Training complete" )
    
    # ------ Saving pkl into file -------------------
    torch.save(G.state_dict(), G_pkl_path)
    torch.save(D.state_dict(), D_pkl_path)

    logger.info("Training Done!")


    
    # ------ Analyzing the results / Plotting ------
    F_Sum, F_x_Mean, F_z_Mean, F_x_Std, F_z_Std = check_fake_data(3000)
    logger.info('Fake data checked!')
    
    # --- Real data check
    R_Sum, R_x_Mean, R_z_Mean, R_x_Std, R_z_Std = load_check_real_data( datafile , 3000)
    logger.info('Real data checked!')
    
    # --- Plotiting
    plot_all_info()
