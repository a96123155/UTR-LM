#!/usr/bin/env python
# coding: utf-8
# 设置CUDA_VISIBLE_DEVICES环境变量，指定在哪个GPU上运行，输出日志到指定文件
# Set the CUDA_VISIBLE_DEVICES environment variable to specify which GPU to run on, and output logs to a specified file.
# CUDA_VISIBLE_DEVICES=3 nohup python v2_ESM2_alldata_SupervisedInfo.py --prefix 3.1 --layers 6 --heads 16 --embed_dim 256 --train_fasta /home/ubuntu/5UTR_Optimizer-master/data/FiveSpecies_Cao_allutr_with_energy.fasta > ./output/ESM2_3.1_fivepeciesCao.out 2>&1 &

# 导入库
# Import libraries
import os
import pathlib
import random
import math
from copy import deepcopy
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchsummary import summary
import torch.nn.functional as F
import argparse
from argparse import Namespace
import esm
from esm.data import *
from esm.model.esm2_supervised import ESM2
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default = '1.1')
parser.add_argument('--cell_line', type=str, default = 'fiveSpeciesCao')
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--layers', type = int, default = 6)
parser.add_argument('--heads', type = int, default = 16)
parser.add_argument('--embed_dim', type = int, default = 256)
parser.add_argument('--supervised_weight', type = float, default = 1)
parser.add_argument('--batch_toks', type = int, default = 4096)
parser.add_argument('--train_fasta', type = str, default = '/home/ubuntu/5UTR_Optimizer-master/data/FiveSpecies_Cao_allutr_with_energy.fasta')
args = parser.parse_args()

# Declare global variables
global idx_to_tok, prefix, epochs, layers, heads, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id

# Set the GPU to use for computation
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set variables for the experiment
prefix = f'ESM2SI_{args.prefix}'
epochs = args.epochs
layers = args.layers
heads = args.heads
embed_dim = args.embed_dim
batch_toks = args.batch_toks
train_fasta = args.train_fasta

# Set variables for the model representation
repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = False

# Create an output file name based on the experiment settings
outputfilename = f'{prefix}_{args.cell_line}_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks_supervisedweight{args.supervised_weight}'
print(outputfilename)

# Create an Alphabet object with specific parameters
alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

# Create a dictionary for mapping indices to tokens
idx_to_tok = dict()
for k,v in alphabet.tok_to_idx.items():
    idx_to_tok[v] = k
idx_to_tok[-1] = '-'
print(idx_to_tok)

# Get the ID for the mask token
mask_toks_id = alphabet.tok_to_idx['<mask>']  

# Load the dataset from a file and create data loader
dataset = FastaBatchedDataset.from_file(train_fasta, mask_prob = 0.15)
batches = dataset.get_batch_indices(toks_per_batch = batch_toks, extra_toks_per_seq = 1)
data_loader = torch.utils.data.DataLoader(
    dataset, 
    collate_fn=alphabet.get_batch_converter(), 
    batch_sampler=batches
)
print(f"{len(dataset)} sequences")
print([len(b) for b in batches])

# Create the ESM2 model with specified parameters
model = ESM2(num_layers = layers,
     embed_dim = embed_dim,
     attention_heads = heads,
     alphabet = alphabet).to(device)
print(model)


# Set up the optimizer for the model
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4,
                             betas=(0.9, 0.98),
                             eps=1e-8)
# Initializes some variables, including the best loss values, the time taken for training, and empty lists to store the loss values.
loss_mlm_best, loss_supervised_best, ep_best = np.inf, np.inf, -1
time_train = 0
loss_mlm_list, loss_supervised_list = [], []

dir_saver = './saved_models/' # 模型训练完后保存模型的路径。
if not os.path.exists(dir_saver): # 如果路径不存在，则创建路径。
    os.makedirs(dir_saver)
    
if not os.path.exists('./figures/'): os.makedirs('./figures/') # 如果保存模型训练过程中的figures的路径不存在，则创建路径。

for epoch in range(1, epochs + 1): # 对于每个epoch进行迭代。
    print(f'{epoch}/{epochs}') # 输出当前训练的epoch数。
    loss_mlm_epoch, loss_supervised_epoch = [], []
    for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(data_loader): # 对于每个batch进行迭代。
        masked_toks = masked_toks.to(device) # 将masked_toks移动到GPU上。
        supervised_labels = torch.FloatTensor([eval(l) for l in labels]).to(device)
        loss_supervised_index = (supervised_labels != 1e-9) # 选取supervised_labels中值不为1e-9的索引。
        supervised_labels = supervised_labels[loss_supervised_index].reshape(-1,1) # 重新调整supervised_labels的形状
        
        if truncate: # https://github.com/facebookresearch/esm/issues/21
            masked_toks = masked_toks[:, -1022:] # 如果truncate为True，则将masked_toks和toks中的元素个数截断到1022个
            toks = toks[:, -1022:]
            
        model.train() # 将模型设置为训练模式
        out = model(masked_toks, repr_layers=repr_layers, return_representation = False, return_contacts=return_contacts,
                   return_attentions_symm = False, return_attentions = False) # 对masked_toks进行前向传播，得到输出
        
        logits = out["logits"] # 得到预测值
            
        label = deepcopy(toks).to(device)
        label.masked_fill_((masked_toks != mask_toks_id), -1) # 将toks进行深拷贝，然后将label中不为<mask>的tokens用-1进行填充
        loss_mlm = F.cross_entropy(logits.transpose(1, 2), label, ignore_index = -1) # 计算多分类的交叉熵损失
        loss_mlm_epoch.append(loss_mlm.cpu().detach().tolist())
        
        if len(supervised_labels) != 0: # 如果supervised_labels不为空，则计算监督信息的均方误差损失，并将两个损失进行加权
            logits_supervised = out["logits_supervised"][loss_supervised_index]
            loss_supervised = F.mse_loss(logits_supervised, supervised_labels)
            loss_supervised_epoch.append(loss_supervised.cpu().detach().tolist())
            
            loss = loss_mlm + args.supervised_weight * loss_supervised
#             print(len(supervised_labels), loss_mlm, loss_supervised)
        else:
            loss = loss_mlm # 如果supervised_labels为空，则只使用交叉熵损失
        
        optimizer.zero_grad() # 将梯度清零
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 根据梯度更新模型参数

    # 在训练过程中，计算每个epoch的MLM loss和Supervised Info loss的平均值，并将它们添加到对应的列表(loss_mlm_list和loss_supervised_list)中
    loss_mlm_epoch = np.mean(loss_mlm_epoch)
    loss_mlm_list.append(loss_mlm_epoch)
    
    loss_supervised_epoch = np.mean(loss_supervised_epoch)
    loss_supervised_list.append(loss_supervised_epoch)

    # 如果当前epoch的MLM loss比之前的最佳值(loss_mlm_best)更小，那么更新loss_mlm_best和ep_best，并将模型的状态字典保存到文件(path_saver)中
    if loss_mlm_epoch < loss_mlm_best: 
        loss_mlm_best, ep_best = loss_mlm_epoch, epoch
        path_saver = dir_saver + f'{outputfilename}_MLMLossMin.pkl'
        print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best} | Train_Loss_Best = {loss_mlm_best:.4f}')
        
        torch.save(model.eval().state_dict(), path_saver)
        
        # 用matplotlib库画出一个包含两个子图的图像，其中第一个子图展示MLM loss的变化情况，第二个子图展示Supervised Info loss的变化情况，并将它们保存到文件'./figures/{outputfilename}.tif'中
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 10))
        axes[0].plot(range(epoch), loss_mlm_list, label = 'MLM|CELoss')
        axes[1].plot(range(epoch), loss_supervised_list, label = 'Supervised Info|MSELoss')
        plt.title(f'{outputfilename}')
        plt.legend()
        plt.savefig(f'./figures/{outputfilename}.tif')


