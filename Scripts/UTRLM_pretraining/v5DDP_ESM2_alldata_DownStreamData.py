#!/usr/bin/env python
# coding: utf-8
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python v5DDP_ESM2_alldata_DownStreamData.py --prefix 3.1 --layers 6 --heads 16 --embed_dim 256 --train_fasta /home/ubuntu/5UTR_Optimizer-master/data/FiveSpecies_Cao_allutr_with_energy.fasta > ./output/ESM2DSD_3.1_fivepeciesCao.out 2>&1 &
# netstat -ntlp | grep 2345
# nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 2345 v2DDP_ESM2_alldata_SupervisedInfo.py --prefix DDP3.2 --lr 0.001 --layers 6 --heads 16 --embed_dim 128 --train_fasta /home/ubuntu/5UTR_Optimizer-master/data/FiveSpecies_Cao_SampleTest_allutr_with_energy.fasta --device_ids 0,1 --load_wholemodel --modelfile /home/ubuntu/esm2/saved_models/ESM2SI_DDP3.2_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr0.001_MLMLossMin.pkl --epochs 150 --init_epochs 73 > ./output/DDP3.2_ESM2.out 2>&1 &


import esm
from esm.data import *
from esm.model.esm2 import ESM2
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

import os
import pathlib
from copy import deepcopy
from tqdm import tqdm, trange

import torch
from torchsummary import summary
import torch.nn.functional as F

import math
import numpy as np
import pandas as pd
import random
random.seed(1337)

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
from argparse import Namespace

from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0,1,2', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")

parser.add_argument('--prefix', type=str, default = 'DDP3.2')
parser.add_argument('--cell_line', type=str, default = 'fiveSpeciesCao')
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--layers', type = int, default = 6)
parser.add_argument('--heads', type = int, default = 16)
parser.add_argument('--embed_dim', type = int, default = 128)
parser.add_argument('--batch_toks', type = int, default = 4096)
parser.add_argument('--train_fasta', type = str, default = '/home/ubuntu/5UTR_Optimizer-master/data/FiveSpecies_Cao_allutr_with_energy.fasta')

parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--load_wholemodel', action = 'store_true') ## if --: True
parser.add_argument('--modelfile', type = str, default = '/home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')
parser.add_argument('--init_epochs', type = int, default = 0)

args = parser.parse_args()

global idx_to_tok, prefix, epochs, layers, heads, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id ,device_ids, device

device_ids = list(map(int, args.device_ids.split(',')))
dist.init_process_group(backend='nccl')
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
torch.cuda.set_device(device)

prefix = f'ESM2DSD_{args.prefix}'
epochs = args.epochs
layers = args.layers
heads = args.heads
embed_dim = args.embed_dim
batch_toks = args.batch_toks
train_fasta = args.train_fasta

repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = False
outputfilename = f'{prefix}_{args.cell_line}_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks_lr{args.lr}'
print(outputfilename)

# In[4]:


alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

idx_to_tok = dict()
for k,v in alphabet.tok_to_idx.items():
    idx_to_tok[v] = k
idx_to_tok[-1] = '-'
print(idx_to_tok)

mask_toks_id = alphabet.tok_to_idx['<mask>']  

# In[6]:


dataset = FastaBatchedDataset.from_file(train_fasta, mask_prob = 0.15)
batches = dataset.get_batch_indices(toks_per_batch = batch_toks, extra_toks_per_seq = 2)
batches_sampler = DistributedSampler(batches, shuffle = True)
batches_loader = torch.utils.data.DataLoader(batches, 
                                             batch_size = 1,
                                             num_workers = 8,
                                             sampler = batches_sampler)
print(f"{len(dataset)} sequences")
print(f'{len(batches)} batches')

model = ESM2(num_layers = layers,
             embed_dim = embed_dim,
             attention_heads = heads,
             alphabet = alphabet).to(device)

if args.load_wholemodel: 
    model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(args.modelfile).items()}) # , strict=False
    
model = DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]], output_device=device_ids[args.local_rank], find_unused_parameters=True)

print(model)

# In[29]:


# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=1e-4,
#                              betas=(0.9, 0.98),
#                              eps=1e-8)

optimizer = torch.optim.SGD(model.parameters(),
                             lr = args.lr, 
                             momentum=0.9,
                             weight_decay = 1e-4)

loss_best, ep_best = np.inf, -1
time_train = 0
loss_list = []

dir_saver = './saved_models/'
if not os.path.exists(dir_saver):
    os.makedirs(dir_saver)
    
if not os.path.exists('./figures/'): os.makedirs('./figures/')

for epoch in range(args.init_epochs+1, args.init_epochs + epochs + 2):
    print(f'{epoch}/{epochs}')
    loss_epoch = []
    
    for i, batch in tqdm(enumerate(batches_loader)):
        batch = np.array(torch.LongTensor(batch)).tolist()
        #print([batch])
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                collate_fn=alphabet.get_batch_converter(), 
                                                batch_sampler=[batch], 
                                                shuffle = False)
        
        for (labels, strs, masked_strs, toks, masked_toks, _) in dataloader:
            
            masked_toks = masked_toks.to(device)

            if truncate: # https://github.com/facebookresearch/esm/issues/21
                masked_toks = masked_toks[:, -1022:]
                toks = toks[:, -1022:]

            model.train()
            out = model(masked_toks, repr_layers=repr_layers, return_representation = True, return_contacts=return_contacts)

            logits = out["logits"]

            label = deepcopy(toks).to(device)
            label.masked_fill_((masked_toks != mask_toks_id), -1)
            loss = F.cross_entropy(logits.transpose(1, 2), label, ignore_index = -1)
            loss_epoch.append(loss.cpu().detach().tolist())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_epoch = np.mean(loss_epoch)
    loss_list.append(loss_epoch)
    print(f'Epoch = {epoch}: MLM Loss = {loss_epoch:.4f}')
    
    if loss_epoch < loss_best: 
        loss_best, ep_best = loss_epoch, epoch
        path_saver = dir_saver + f'{outputfilename}_MLMLossMin_epoch{epoch}.pkl'
        print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best} | Train_Loss_Best = {loss_best:.4f}')
        
        torch.save(model.eval().state_dict(), path_saver)
        
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 10))
        axes[0].plot(range(epoch-args.init_epochs), loss_list, label = 'MLM|CELoss')
        plt.title(f'{outputfilename}')
        plt.legend()
        plt.savefig(f'./figures/{outputfilename}.tif')



