#!/usr/bin/env python
# coding: utf-8
# netstat -ntlp | grep 2345
# nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 2345 v3DDP_ESM2_alldata_SupervisedInfo_SecondaryStructure.py --prefix DDP4.2 --lr 1e-5 --layers 6 --heads 16 --embed_dim 128 --train_fasta /home/ubuntu/human_5utr_modeling-master/data/FiveSpecies_Cao_allutr_with_energy_structure.fasta --device_ids 3,2,1,0 > ./output/DDP5.2_ESM2_1e-5.out 2>&1 &


import esm
from esm.data import *
from esm.model.esm2_secondarystructure import ESM2
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

parser.add_argument('--prefix', type=str, default = 'DDP5.1')
parser.add_argument('--cell_line', type=str, default = 'fiveSpeciesCao')
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--layers', type = int, default = 6)
parser.add_argument('--heads', type = int, default = 16)
parser.add_argument('--embed_dim', type = int, default = 128)
parser.add_argument('--batch_toks', type = int, default = 4096)
parser.add_argument('--supervised_weight', type = float, default = 1)
parser.add_argument('--structure_weight', type = float, default = 1)
parser.add_argument('--train_fasta', type = str, default = '/home/ubuntu/human_5utr_modeling-master/data/FiveSpecies_Cao_allutr_with_energy_structure.fasta')

parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--load_wholemodel', action = 'store_true') ## if --: True
parser.add_argument('--modelfile', type = str, default = '/home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')
parser.add_argument('--init_epochs', type = int, default = 0)

args = parser.parse_args()

global idx_to_tok, prefix, epochs, layers, heads, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id ,device_ids, device, structure_tok_to_idx

device_ids = list(map(int, args.device_ids.split(','))) # 指定设备ID
dist.init_process_group(backend='nccl') # 初始化进程组，用于启用分布式训练
device = torch.device('cuda:{}'.format(device_ids[args.local_rank])) # 使用 args.local_rank 确定当前进程使用的设备，并将其设置为默认设备
torch.cuda.set_device(device)

# 设置一些模型超参数，如层数、头数、嵌入维度和批处理大小，并使用这些参数生成输出文件名 outputfilename
prefix = f'ESM2SISS_{args.prefix}'
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
outputfilename = f'{prefix}_{args.cell_line}_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks_lr{args.lr}_supervisedweight{args.supervised_weight}_structureweight{args.structure_weight}'
print(outputfilename)

# 创建一个字典 structure_tok_to_idx，用于将二级结构符号转换为索引
structure_tok_to_idx = {'(': 0, '.': 1, ')': 2}

# 创建一个 Alphabet 实例，用于将原始序列编码为数字。通过将 mask_prob 参数设置为 0.15，可以随机屏蔽一些氨基酸以进行掩蔽训练
alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

# 将字典 alphabet.tok_to_idx 中的键-值对反转，以获得index-tok的映射关系，并将 -1 的tok设置为 -
idx_to_tok = dict()
for k,v in alphabet.tok_to_idx.items():
    idx_to_tok[v] = k
idx_to_tok[-1] = '-'
print(idx_to_tok)

mask_toks_id = alphabet.tok_to_idx['<mask>']  

# 从 fasta 文件中创建数据集，每个序列的氨基酸将根据 mask_prob 参数进行随机掩蔽。然后，使用 get_batch_indices() 方法将数据集划分为多个批次
dataset = FastaBatchedDataset.from_file(train_fasta, mask_prob = 0.15)
batches = dataset.get_batch_indices(toks_per_batch = batch_toks, extra_toks_per_seq = 2)
batches_sampler = DistributedSampler(batches, shuffle = True) # 使用 DistributedSampler 创建一个 batches_sampler，将批次随机化并将其分配给多个进程
batches_loader = torch.utils.data.DataLoader(batches, 
                                             batch_size = 1,
                                             num_workers = 8,
                                             sampler = batches_sampler) # 使用 DataLoader 将 batches 装载到内存中，以便能够随机访问
print(f"{len(dataset)} sequences")
print(f'{len(batches)} batches')
# 使用 ESM2 类定义模型，并使用 DistributedDataParallel 对其进行封装，以便在多个 GPU 上进行分布式训练
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

loss_mlm_best, loss_supervised_best, loss_structure_best, ep_best = np.inf, np.inf, np.inf, -1
time_train = 0
loss_mlm_list, loss_supervised_list, loss_structure_list = [], [], []

dir_saver = './saved_models/'
if not os.path.exists(dir_saver):
    os.makedirs(dir_saver)
    
if not os.path.exists('./figures/'): os.makedirs('./figures/')
# 外层循环是对训练的 epoch 进行循环，内层循环是对每个 batch 进行循环
for epoch in range(args.init_epochs+1, args.init_epochs + epochs + 1):
    print(f'{epoch}/{args.init_epochs + epochs + 1}')
    loss_mlm_epoch, loss_supervised_epoch, loss_structure_epoch = [], [], []
    
    for i, batch in tqdm(enumerate(batches_loader)):
        # 首先将 batch 转化为 dataloader，然后对 dataloader 中的每个 sample 进行前向计算和反向传播。
        batch = np.array(torch.LongTensor(batch)).tolist()
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                collate_fn=alphabet.get_batch_converter(), 
                                                batch_sampler=[batch], 
                                                shuffle = False)
        
        for (labels, strs, masked_strs, toks, masked_toks, _) in dataloader:
            
            masked_toks = masked_toks.to(device)
            supervised_labels = torch.FloatTensor([eval(l.split('|')[-1]) for l in labels]).to(device)
#             print(supervised_labels)
            structure_labels = [l.split('|')[1] for l in labels]
            structure_labels = [[structure_tok_to_idx[tok] for tok in structure] for structure in structure_labels]
            max_len = masked_toks.shape[1]
            structure_labels = [structure + [-1]*(max_len-len(structure)) for structure in structure_labels]
#             print(structure_labels)
            structure_labels = torch.LongTensor(structure_labels).to(device)
#             print(f'***structure_labels.shape = {structure_labels.shape}')
#             print(f'***supervised_labels.shape = {supervised_labels.shape}')
            
            loss_supervised_index = (supervised_labels != 1e-9)
            supervised_labels = supervised_labels[loss_supervised_index].reshape(-1,1)

            if truncate: # https://github.com/facebookresearch/esm/issues/21
                masked_toks = masked_toks[:, -1022:]
                toks = toks[:, -1022:]
                structure_labels = structure_labels[:, -1022:]
            # 在前向计算中，模型会接收 masked_toks 作为输入，生成 logits 和 structure_logits 作为输出。其中，logits 是用于训练语言模型的输出，而 structure_logits 是用于训练结构预测的输出。
            model.train()
            out = model(masked_toks, repr_layers=repr_layers, return_representation = True, return_contacts=return_contacts,
                       return_attentions_symm = False, return_attentions = False)

            logits = out["logits"] # (Batchsize, seq_length, embed_dim)
            structure_logits = out['logits_structure']
#             print(f'****Logits = {logits.shape} | struc logits = {structure_logits.shape}')
            # 在反向传播中，将 logits，structure_logits 和 supervised_labels（如果有的话）作为参数，计算出损失，并执行梯度下降算法进行参数更新。
            # 每个 epoch 训练完成后，会计算并输出该 epoch 的平均损失（包括 MLM Loss、Supervised Loss 和 Structure Loss）。
            label = deepcopy(toks).to(device)
            label.masked_fill_((masked_toks != mask_toks_id), -1)
            loss_mlm = F.cross_entropy(logits.transpose(1, 2), label, ignore_index = -1)
            loss_mlm_epoch.append(loss_mlm.cpu().detach().tolist())
            
            loss_structure = F.cross_entropy(structure_logits.transpose(1, 2), structure_labels, ignore_index = -1)
            loss_structure_epoch.append(loss_structure.cpu().detach().tolist())
            
#             print(f'***logits.shape = {logits.shape} | structure_logits.shape = {structure_logits.shape}')
#             print(f'***logits.transpose(1, 2).shape = {logits.transpose(1, 2).shape} | label.shape = {label.shape}')
            #print(loss_mlm)
            if len(supervised_labels) != 0: 
                logits_supervised = out["logits_supervised"][loss_supervised_index]
                loss_supervised = F.mse_loss(logits_supervised, supervised_labels)
                loss_supervised_epoch.append(loss_supervised.cpu().detach().tolist())
#                 print(f'Epoch = {epoch}: MLM Loss = {loss_mlm:.4f} | Supervised Loss = {loss_supervised_epoch:.4f}')
                loss = loss_mlm + args.supervised_weight * loss_supervised + args.structure_weight * loss_structure
            else:
                loss = loss_mlm + args.structure_weight * loss_structure
#                 print(f'Epoch = {epoch}: MLM Loss = {loss_mlm:.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_mlm_epoch = np.mean(loss_mlm_epoch)
    loss_mlm_list.append(loss_mlm_epoch)
    
    loss_supervised_epoch = np.mean(loss_supervised_epoch)
    loss_supervised_list.append(loss_supervised_epoch)
    
    loss_structure_epoch = np.mean(loss_structure_epoch)
    loss_structure_list.append(loss_structure_epoch)
    print(f'Epoch = {epoch}: MLM Loss = {loss_mlm_epoch:.4f} | Supervised Loss = {loss_supervised_epoch:.4f} | Structure Loss = {loss_structure_epoch:.4f}')
    
    if loss_mlm_epoch < loss_mlm_best: 
        loss_mlm_best, ep_best = loss_mlm_epoch, epoch
        path_saver = dir_saver + f'{outputfilename}_MLMLossMin_epoch{epoch}.pkl'
        print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best} | Train_Loss_Best = {loss_mlm_best:.4f}')
        
        torch.save(model.eval().state_dict(), path_saver)
        
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 10))
        axes[0].plot(range(epoch-args.init_epochs), loss_mlm_list, label = 'MLM|CELoss')
        axes[1].plot(range(epoch-args.init_epochs), loss_supervised_list, label = 'Supervised Info|MSELoss')
        axes[2].plot(range(epoch-args.init_epochs), loss_structure_list, label = 'Structure|CELoss')
        plt.title(f'{outputfilename}')
        plt.legend()
        plt.savefig(f'./figures/{outputfilename}.tif')



