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
from models.2Dmodel import *
import uproot
import random



def train_D(D, device, data, d_optimizer, epoch, batchsize, noise_dim):
        d_f_loss,d_r_loss = 0,0
        criterion = nn.BCELoss()
        data = torch.Tensor(data).to(device).requires_grad_()
        d_optimizer.zero_grad()
        d_r_score = D(data)
        d_r_standard = torch.ones(batchsize,1)
        d_r_loss = criterion( d_r_score, d_r_standard.to(device) )  # ones = true
        d_r_loss.backward()
        noise = noiseSam(batchsize, noise_dim )
        fake_data = G( noise.to(device) )
        d_f_score = D( fake_data )
        d_f_standard = torch.zeros(batchsize,1)         
        d_f_loss = criterion( d_f_score, d_f_standard.to(device))  # zeros = fake
        d_f_loss.backward()
        d_optimizer.step()
        return d_r_loss+d_f_loss

def train_G(G, device, real_data, g_optimizer, epoch, batchsize, noise_dim, feature_matching):
        noise = noiseSam(batchsize, noise_dim)
        fake_data = G( noise.to(device) )
        g_score = D( fake_data )
        g_standard = torch.ones(g_minibatch_size,1)
        g_loss = criterion( g_score, g_standard.to(device) )
        if feature_matching:
            r_feature, _       = D( real_data,matching=True )
            f_feature, g_score = D( fake_data,matching=True )
            r_feature = torch.mean(r_feature,0)
            f_feature = torch.mean(f_feature,0)
            criterionG = nn.MSELoss()
            g_loss_FM = criterionG(f_feature,r_feature)
            g_loss = g_loss + g_loss_FM
        g_loss.backward()
        g_optimizer.step()
        return g_loss
        

def get_parser():
        parser = argparse.ArgumentParser(
            description='Run histograms training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--datafile', action='store', type=str,
                                                help='ROOT file paths')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                                                help='number of epochs to train (default: 50)')
        parser.add_argument('--batchsize', type=int, default=25,
                                                help='size of Batch (default: 25)')
        parser.add_argument('--num_hist', action='store', type=int, default=500,
                                                help='number of histograms that loads')
        parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                                                help='learning rate (default: 0.0000001)')
        parser.add_argument('--lr2', type=float, default=1e-5, metavar='LR',
                                                help='learning rate2 (default: 0.00001)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                                help='SGD momentum (default: 0.5)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                                                help='random seed (default: 1)')
        parser.add_argument('--model_pt_path', action='store', type=str,
                                                help='Model pt path file paths')
        parser.add_argument('--model_restore_pt_path', action='store', type=str,
                                                help='Model restore pt path file paths')
        parser.add_argument('--D_pt_path', action='store', type=str,
                                                help='D Model pt path file paths')
        parser.add_argument('--D_restore_pt_path', action='store', type=str,
                                                help='D Model restore pt path file paths')
        parser.add_argument('--restore', action='store', type=ast.literal_eval, default=False,
                                                help='ckpt file paths')
        return parser


def main():
        logger.info("Start...")
        parser = get_parser()
        parse_args = parser.parse_args()

        datafile = parse_args.datafile
        num_epochs = parse_args.epochs
        learning_rate  = parse_args.lr
        learning_rate2 = parse_args.lr2
        momentum = parse_args.momentum
        seed = parse_args.seed
        model_pt_path = parse_args.model_pt_path
        model_restore_pt_path = parse_args.model_restore_pt_path
        D_pt_path = parse_args.D_pt_path
        D_restore_pt_path = parse_args.D_restore_pt_path
        num_hist = parse_args.num_hist
        batchsize = parse_args.batchsize
        restore = parse_args.restore

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
        Dataset_Test = HistoDataset(datafile,num_hist)
        test_loader = torch.utils.data.DataLoader(Dataset_Test, batch_size=batchsize,
                                                shuffle=False, num_workers=0)
        logger.info('Load data successfully! Start training...')

        # first try: only train VAE
        model = CVAE().to(device)
        D = DNet().to(device)
        if restore:
                model.load_state_dict(torch.load(model_restore_pt_path))
                D.load_state_dict(torch.load(D_restore_pt_path))
                logger.info('Load model from:')
                logger.info(model_restore_pt_path)
                logger.info(D_restore_pt_path)
        #UNet = AttU_Net3D().to(device)

        
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        optimizer  = optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.5, 0.999))
        optimizer2 = optim.Adam(model.parameters(), lr=learning_rate2, betas=(0.5, 0.999))
        #optimizer2 = optim.SGD(UNet.parameters(), lr=args.lr, momentum=args.momentum)
        
        for epoch in range( 1,num_epochs+1 ):
                #train(args, model, device, train_loader, optimizer, epoch, UNet, optimizer2)
                #test(args, model, device, test_loader, epoch, UNet)
                train(model, D, device, train_loader, optimizer, optimizer2, epoch, batchsize)
                losses,losses_MSE,losses_BCE = test(model, D, device, test_loader, epoch, batchsize)
                
                if epoch == 1:
                        logger.info("1 epoch completed! This code is running successfully!")
                if epoch%(num_epochs//10)==0:
                        logger.info( "Epoch %6d. Loss %5.3f. Loss_MSE %5.3f. Loss_BCE %5.3f." % ( epoch, losses, losses_MSE, losses_BCE ) )
                if epoch%200==0:
                        torch.save(model.state_dict(), model_pt_path)
                        torch.save(D.state_dict(), D_pt_path)
                        logger.info('Model save into:')
                        logger.info(model_pt_path)
                        logger.info(D_pt_path)

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
