from __future__ import print_function
import argparse
from sklearn.utils import shuffle
import sys
import argparse
import logging
import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import Dataset
from dataset import *
from models.model2D import *
import uproot
import random

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()

def train_D(D, G, noiseSam, device, data, d_optimizer, epoch, batchsize, noise_dim, Minidisc):
        d_f_loss,d_r_loss = 0,0
        criterion = nn.BCELoss()
        d_optimizer.zero_grad()
        d_r_score = D(data)
        d_r_standard = torch.ones(batchsize,1)
        d_r_loss = criterion( d_r_score, d_r_standard.to(device) )  # ones = true
        d_r_loss.backward()
        noise = noiseSam(batchsize, noise_dim )
        fake_data = G( noise.to(device) )
        d_f_score = D( fake_data,minidisc=Minidisc )
        d_f_standard = torch.zeros(batchsize,1)
        d_f_loss = criterion( d_f_score, d_f_standard.to(device))  # zeros = fake
        d_f_loss.backward()
        d_optimizer.step()
        return d_r_loss+d_f_loss

def train_G(D, G, noiseSam, device, real_data, g_optimizer, epoch, batchsize, noise_dim, feature_matching, Minidisc):
        g_loss,g_loss_FM=0,0
        criterion = nn.BCELoss()
        noise = noiseSam(batchsize, noise_dim)
        fake_data = G( noise.to(device) )
        if feature_matching:
            r_feature, _       = D( real_data,matching=True,minidisc=Minidisc )
            f_feature, g_score = D( fake_data,matching=True,minidisc=Minidisc )
            r_feature = torch.mean(r_feature,0)
            f_feature = torch.mean(f_feature,0)
            criterionG = nn.MSELoss()
            g_loss_FM = criterionG(f_feature,r_feature)
            g_loss = g_loss + g_loss_FM
        else:
            g_score = D(fake_data,minidisc=Minidisc)
            g_standard = torch.ones(batchsize,1)
            g_loss = criterion( g_score, g_standard.to(device) )
        g_loss.backward()
        g_optimizer.step()
        return g_loss


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
    parser.add_argument('--num_hist', action='store', type=int, default=500,
                        help='Number of histograms to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=25,
                        help='batchsize')
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
    parser.add_argument('--wgan', action='store', type=ast.literal_eval, default=False,
                        help='w-GAN')
    parser.add_argument('--gradient_penalty', action='store', type=ast.literal_eval, default=False,
                        help='gp')
    parser.add_argument('--feature_matching', action='store', type=ast.literal_eval, default=False,
                        help='fm')
    parser.add_argument('--minidisc', action='store', type=ast.literal_eval, default=False,
                        help='md')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    return parser


def main():
        logger.info("Start...")
        parser = get_parser()
        parse_args = parser.parse_args()
        num_epochs = parse_args.num_epochs
        num_hist = parse_args.num_hist
        D_restore_pkl_path = parse_args.D_restore_pkl_path
        G_restore_pkl_path = parse_args.G_restore_pkl_path
        G_pkl_path = parse_args.G_pkl_path
        D_pkl_path = parse_args.D_pkl_path
        datafile = parse_args.datafile
        outfile = parse_args.outfile
        seed = parse_args.seed
        wgan = parse_args.wgan
        restore = parse_args.restore
        gradient_penalty = parse_args.gradient_penalty
        feature_matching = parse_args.feature_matching
        minidisc = parse_args.minidisc
        d_learning_rate = parse_args.d_lr
        g_learning_rate = parse_args.g_lr
        batchsize = parse_args.batchsize
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
        logger.info('constructing graph')

        #Set Random Seed
        torch.manual_seed(seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(device)

        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        #load the data.
        logger.info('Start loading data:')
        logger.info(datafile)
        Dataset_Train = HistoDataset(datafile,num_hist)
        train_loader = torch.utils.data.DataLoader(Dataset_Train, batch_size=batchsize,
                                                shuffle=True, num_workers=0)
        logger.info('Load data successfully! Start training...')

        # first try: only train VAE
        noiseSam=get_noise_sampler()
        D = Discriminator_4L().to(device)
        G = Generator_4L(noise_dim).to(device)
        if restore:
                G.load_state_dict(torch.load(G_restore_pkl_path))
                D.load_state_dict(torch.load(D_restore_pkl_path))
                print('restored from ',G_restore_pkl_path)
                print('restored from ',D_restore_pkl_path)
        
        d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
        
        for epoch in range( 1,num_epochs+1 ):
                for batch_idx, data in enumerate(train_loader):
                        data = torch.Tensor(data).to(device).requires_grad_()
                        for i in range(2):
                            d_loss = train_D(D, G, noiseSam, device, data, d_optimizer, epoch, batchsize, noise_dim, minidisc)
                        g_loss = train_G(D, G, noiseSam, device, data, g_optimizer, epoch, batchsize, noise_dim, feature_matching, minidisc)
                if epoch == 1:
                        logger.info("1 epoch completed! This code is running successfully!")
                if epoch%(num_epochs//10)==0:
                        logger.info( "Epoch %6d. D_Loss %5.3f. G_Loss %5.3f." % ( epoch, d_loss, g_loss ) )
                if epoch%200==0:
                        torch.save(D.state_dict(), D_pkl_path)
                        torch.save(G.state_dict(), G_pkl_path)
                        logger.info('Model save into:')
                        logger.info(D_pkl_path)
                        logger.info(G_pkl_path)
        logger.info("Train Done!")

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)

if __name__ == '__main__':
    main()
