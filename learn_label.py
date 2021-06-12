from __future__ import print_function
import argparse
from sklearn.utils import shuffle
import sys
import argparse
import logging
import os
import ast
import gc
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import Dataset
from datasets.dataset_label import *
from models.model_label import *
import uproot
import random

def get_noise_sampler(): # rand:uniform  randn:normal
    return lambda m, n: torch.randn(m, n).requires_grad_()

def train_D(D, G, noiseSam, device, data, d_optimizer, epoch, batchsize, noise_dim, Minidisc):
        d_optimizer.zero_grad()
        d_r_score = D(data)
        noise = noiseSam( batchsize, noise_dim )
        fake_data = G( noise.to(device) )
        d_f_score = D( fake_data )
        #print("d_f_score:",d_f_score)
        # gradient panelty
        alpha = torch.randn((batchsize,1))
        alpha = alpha.cuda()
        #print("alpha=",alpha.shape)
        #print("data=",data.shape)
        #print("fakedata=",fake_data.shape)
        hat_data = alpha*data + (1-alpha)*fake_data #sampling zone
        hat_data.cuda().requires_grad_()
        d_hat_score = D(hat_data)

        gradients = torch.autograd.grad(outputs=d_hat_score, inputs=hat_data,
                    grad_outputs=torch.ones(d_hat_score.size()).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        lambda_ = 10
        gp = lambda_*((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        d_loss = -torch.mean(d_r_score) + torch.mean(d_f_score) + gp

        d_loss.backward()

        d_optimizer.step()
        return d_loss

def train_G(D, G, noiseSam, device, real_data, g_optimizer, epoch, batchsize, noise_dim, feature_matching, Minidisc):
        g_optimizer.zero_grad()
        noise = noiseSam(batchsize, noise_dim)
        fake_data = G( noise.to(device) )
        g_score = D(fake_data)
        #print("g_score:",g_score)
        g_loss = -torch.mean(g_score)
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()

def train_D_gan(D, G, noiseSam, device, data, d_optimizer, epoch, batchsize, noise_dim, Minidisc):
        criterion = nn.BCELoss()
        d_optimizer.zero_grad()
        d_r_score = D(data)
        #print("d_r_score:",d_r_score)
        d_r_standard = torch.ones(batchsize,1)
        d_r_loss = criterion( d_r_score, d_r_standard.to(device) )  # ones = true
        d_r_loss.backward()
        noise = noiseSam( batchsize, noise_dim )
        fake_data = G( noise.to(device) )
        d_f_score = D( fake_data )
        #print("d_f_score:",d_f_score)
        d_f_standard = torch.zeros(batchsize,1)
        d_f_loss = criterion(d_f_score, d_f_standard.to(device))  # zeros = fake
        d_f_loss.backward()
        d_optimizer.step()
        return d_r_loss+d_f_loss

def train_G_gan(D, G, noiseSam, device, real_data, g_optimizer, epoch, batchsize, noise_dim, feature_matching, Minidisc):
        criterion = nn.BCELoss()
        g_optimizer.zero_grad()
        noise = noiseSam(batchsize, noise_dim)
        fake_data = G( noise.to(device) )
        g_score = D(fake_data)
        #print("g_score:",g_score)
        g_standard = torch.ones(batchsize,1)
        g_loss = criterion( g_score, g_standard.to(device) )
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()


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
    parser.add_argument('--hidden_size', action='store', type=int, default=400,
                        help='hidden layer dimention')
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
        hidden_size = parse_args.hidden_size
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(device)

        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        noiseSam=get_noise_sampler()
        D = Discriminator(hidden_size).to(device)
        G = Generator(noise_dim,hidden_size).to(device)
        if restore:
                G.load_state_dict(torch.load(G_restore_pkl_path))
                D.load_state_dict(torch.load(D_restore_pkl_path))
                print('restored from ',G_restore_pkl_path)
                print('restored from ',D_restore_pkl_path)
        
        d_optimizer = optim.RMSprop(D.parameters(), lr=d_learning_rate)
        g_optimizer = optim.RMSprop(G.parameters(), lr=g_learning_rate)
       
        #num_load = 2
 
        for epoch in range( 1,num_epochs+1 ):
                #if epoch%(num_epochs//num_load)==0 or epoch==1:
                if epoch==1:
                    #load the data.
                    logger.info('Start loading data:')
                    logger.info(datafile)
                    Dataset_Train, train_loader = None, None
                    del Dataset_Train, train_loader
                    gc.collect()
                    Dataset_Train = HistoDataset(datafile,num_hist)
                    train_loader = torch.utils.data.DataLoader(Dataset_Train, batch_size=batchsize,
                                                               shuffle=True, num_workers=0)
                    logger.info('Load data successfully! Start training...')
                    cpu_info = psutil.virtual_memory()
                    logger.info('memory usage %.2f(GB); total memory %.1f(GB); percent %.1f; cpu count %d.' %(psutil.Process(os.getpid()).memory_info().rss/(1024**3), cpu_info.total/(1024**3) , cpu_info.percent , psutil.cpu_count() ))
                    logger.info('memory_allocated:%.2f Mb, max_memory_allocated:%.2f Mb, memory_cached:%.2f Mb, max_memory_cached:%.2f Mb.' %(float(torch.cuda.memory_allocated()/1024**2),float(torch.cuda.max_memory_allocated()/1024**2), float(torch.cuda.memory_reserved()/1024**2), float(torch.cuda.max_memory_reserved()/1024**2) ))

                for batch_idx, data in enumerate(train_loader):
                        data = torch.Tensor(data).to(device).requires_grad_()
                        for i in range(1):
                            d_loss = train_D_gan(D, G, noiseSam, device, data, d_optimizer, epoch, batchsize, noise_dim, minidisc)
                        for i in range(1):
                            g_loss = train_G_gan(D, G, noiseSam, device, data, g_optimizer, epoch, batchsize, noise_dim, feature_matching, minidisc)
                if epoch == 1:
                        logger.info("1 epoch completed! This code is running successfully!")
                if epoch%(num_epochs//10)==0:
                        logger.info( "Epoch %6d. D_Loss %5.3f. G_Loss %5.3f." % ( epoch, d_loss, g_loss ) )
                if epoch%10==0:
                        torch.save(D.state_dict(), D_pkl_path)
                        #torch.save(G.state_dict(), G_pkl_path)
                        G = G.eval()
                        example = noiseSam( 1, noise_dim )
                        resG = torch.jit.trace(G, example.to(device) )
                        resG.save(G_pkl_path)
                        logger.info('Model save into:')
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
