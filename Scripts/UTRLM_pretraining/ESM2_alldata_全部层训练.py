#!/usr/bin/env python
# coding: utf-8
# CUDA_VISIBLE_DEVICES=3 nohup python ESM2_alldata_å¨é¨åè­ç.py --prefix 1.6 --layers 3 --heads 16 --embed_dim 128 --train_fasta /home/ubuntu/esm2/data/five_species_transcript_5UTR_clean_30_1022.fasta > ./output/ESM2_1.6_fivespecies.out 2>&1 &

# 导入所需的库，例如 esm、torch、numpy、pandas 等
import os
import math
import pathlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from tqdm import tqdm, trange
from torchsummary import summary
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from esm.data import *
from esm.model.esm2 import ESM2

random.seed(1337)

# 定义命令行参数解析器（argparse.ArgumentParser），用于设置模型的参数，如层数、注意力头、嵌入维度等。
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default = '1.1')
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--layers', type = int, default = 6)
parser.add_argument('--heads', type = int, default = 20)
parser.add_argument('--embed_dim', type = int, default = 360)
parser.add_argument('--gpu', type = str, default = '0')
parser.add_argument('--batch_toks', type = int, default = 4096)
parser.add_argument('--train_fasta', type = str, default = '/home/ubuntu/esm2/data/Sample_EnsemblHuman_utrs.fasta')
parser.add_argument('--evaluation', type = str, default = 'False')
parser.add_argument('--test_fasta', type = str, default = '/home/ubuntu/esm2/data/Sample_test_utrs.fasta')
args = parser.parse_args()

# 设置 GPU 设备以及相关的环境变量。
global idx_to_tok, prefix, epochs, layers, heads, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prefix = f'ESM2_{args.prefix}'
epochs = args.epochs
layers = args.layers
heads = args.heads
embed_dim = args.embed_dim
batch_toks = args.batch_toks
train_fasta = args.train_fasta

evaluation = args.evaluation
test_fasta = args.test_fasta

repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = False
return_representation = False


# 定义函数 diff2seq，用于比较真实序列和预测序列，并返回正确匹配的 token 数。
def diff2seq(y_true, y_pred, if_print = False):
    if if_print:
        print('Input seq: ', y_true)
        print('Output seq:', y_pred)
    n, m = 0, 0
    for i, j in zip(y_true, y_pred):
        if i == j: n += 1
        if j != '-': m += 1
    if if_print:
        if m != len(y_pred): 
            print(f'Accurate Masked Tokens: {n}/{m}')
        else:
            print(f'Accurate All Tokens: {n}/{m}')
    return f'{n}/{m}'


# 初始化 Alphabet 类，设置蛋白质序列的字母表和掩码概率。

alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

idx_to_tok = dict()
for k,v in alphabet.tok_to_idx.items():
    idx_to_tok[v] = k
idx_to_tok[-1] = '-'
print(idx_to_tok)

mask_toks_id = alphabet.tok_to_idx['<mask>']  

# 从 FASTA 文件中加载训练数据集，并使用 PyTorch DataLoader 对数据进行批处理。

dataset = FastaBatchedDataset.from_file(train_fasta, mask_prob = 0.15)
batches = dataset.get_batch_indices(toks_per_batch = batch_toks, extra_toks_per_seq = 1)
data_loader = torch.utils.data.DataLoader(
    dataset, 
    collate_fn=alphabet.get_batch_converter(), 
    batch_sampler=batches
)
print(f"{len(dataset)} sequences")
print([len(b) for b in batches])


# 如果设置了评估标志，则从另一个 FASTA 文件中加载测试数据集，并创建一个 DataLoader 对象。
if evaluation == 'True':
    test_dataset = FastaBatchedDataset.from_file(test_fasta, mask_prob = 0.15)
    test_batches = test_dataset.get_batch_indices(toks_per_batch = batch_toks, extra_toks_per_seq = 1)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        collate_fn=alphabet.get_batch_converter(), 
        batch_sampler=test_batches
    )
    print(f"{len(test_dataset)} sequences")
    print([len(b) for b in test_batches])

# 定义 ESM2 模型，并将其加载到 GPU 设备上。
model = ESM2(num_layers = layers,
     embed_dim = embed_dim,
     attention_heads = heads,
     alphabet = alphabet).to(device)

# 定义 eval_step 函数，用于在给定的测试数据加载器上评估模型。

def eval_step(model, test_dataloader):
    
    loss_list = []
    y_true_seq_list = []
    y_pred_seq_list, y_pred_masked_seq_list = [], []

    model.eval()
    with torch.no_grad():
        for (_, strs, masked_strs, toks, masked_toks, _) in tqdm(test_dataloader):
            masked_toks = masked_toks.to(device)

            if truncate: # https://github.com/facebookresearch/esm/issues/21
                masked_toks = masked_toks[:, -1022:]
                toks = toks[:, -1022:]
            out = model(masked_toks, return_representation = False, repr_layers=repr_layers, return_contacts=False)
            logits = out["logits"]
            
            y_pred = torch.argmax(logits, axis = 2)
            y_pred_masked = deepcopy(y_pred)
            y_pred_masked.masked_fill_((masked_toks != mask_toks_id), -1)
            y_pred = y_pred.cpu().detach()
            y_pred_masked = y_pred_masked.cpu().detach()
            y_pred_seq = [''.join([idx_to_tok[int(i)] for i in pred[1:-1]]) for pred in y_pred]
            y_pred_masked_seq = [''.join([idx_to_tok[int(i)] for i in pred[1:-1]]) for pred in y_pred_masked]

            label = deepcopy(toks).to(device)
            label.masked_fill_((masked_toks != mask_toks_id), -1)
            
            loss = F.cross_entropy(logits.transpose(1, 2), label, ignore_index = -1, reduction = 'sum')
            loss_list.append(loss.cpu().detach().tolist())
            
            y_true_seq_list.extend(strs)
            y_pred_seq_list.extend(y_pred_seq)
            y_pred_masked_seq_list.extend(y_pred_masked_seq)

    loss = np.mean(loss_list)

    frac_all_tokens, frac_masked_tokens = [], []
    for i in trange(len(y_true_seq_list)):
        y_true = y_true_seq_list[i]
        y_pred = y_pred_seq_list[i]
        y_pred_masked = y_pred_masked_seq_list[i]
        frac_all_tokens.append(diff2seq(y_true, y_pred))
        frac_masked_tokens.append(diff2seq(y_true, y_pred_masked))

    mlm_results = pd.DataFrame([frac_all_tokens, frac_masked_tokens, y_true_seq_list, 
                                y_pred_seq_list, y_pred_masked_seq_list], 
                               index = ['Accurate_num_all_tokens', 'Accurate_num_masked_tokens', 
                                        'input_seq', 'output_seq', 'output_masked_tokens']).T
    
    mlm_results['Accurate_ratio_all_tokens'] = [eval(i) for i in mlm_results.Accurate_num_all_tokens]
    mlm_results['Accurate_ratio_masked_tokens'] = [eval(i) for i in mlm_results.Accurate_num_masked_tokens]
    
    print('***ESM1b MLM task for test dataset (Masked rate = 0.15)***')
    print(f'All tokens: Accurate of {len(mlm_results)} samples (MEAN +/- STD): {mlm_results.Accurate_ratio_all_tokens.mean():.2f} +/- {mlm_results.Accurate_ratio_all_tokens.std():.2f}')
    print(f'Masked tokens: Accurate of {len(mlm_results)} samples (MEAN +/- STD): {mlm_results.Accurate_ratio_masked_tokens.mean():.2f} +/- {mlm_results.Accurate_ratio_masked_tokens.std():.2f}')
    
    return loss, mlm_results


# 初始化优化器（这里使用 Adam 优化器）。

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.0004,
                             betas=(0.9, 0.98),
                             eps=1e-8,
                             weight_decay = 0.01)

loss_train_best, loss_test_best, ep_best = np.inf, np.inf, -1
time_train = 0
loss_train_list, loss_test_list = [], []

dir_saver = './saved_models/'
if not os.path.exists(dir_saver):
    os.makedirs(dir_saver)
    
if not os.path.exists('./figures/'): os.makedirs('./figures/')

# 设置模型训练的循环，每次迭代都会在训练数据加载器上进行一次完整的前向传播和反向传播。同时，它还计算损失，并根据需要更新模型权重。
for epoch in range(1, epochs + 1):
    print(f'{epoch}/{epochs}')
    loss_train = []
    for (_, strs, masked_strs, toks, masked_toks, _) in tqdm(data_loader):
        masked_toks = masked_toks.to(device)
        
        if truncate: # https://github.com/facebookresearch/esm/issues/21
            masked_toks = masked_toks[:, -1022:]
            toks = toks[:, -1022:]
            
        model.train()
        out = model(masked_toks, return_representation = False, repr_layers=repr_layers, return_contacts=False)
        logits = out["logits"]
        label = deepcopy(toks).to(device)
        label.masked_fill_((masked_toks != mask_toks_id), -1)
        loss = F.cross_entropy(logits.transpose(1, 2), label, ignore_index = -1, reduction = 'sum')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_train.append(loss.cpu().detach().tolist())

    loss_train = np.mean(loss_train)
    loss_train_list.append(loss_train)

    # 在每个周期结束时，保存训练损失最低的模型，并在需要时评估测试数据集。
    if evaluation == 'True':
        loss_test, mlm_results = eval_step(model, test_dataloader)
        loss_test_list.append(loss_test)
        
        if loss_test < loss_test_best: 
            loss_test_best, ep_best = loss_test, epoch
            path_saver = dir_saver + f'{prefix}_TestLossMin_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.pkl'
            print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best} | Test_Loss_Best = {loss_test_best:.4f}')
            torch.save(model.eval().state_dict(), path_saver)

    if loss_train < loss_train_best: 
        loss_train_best, ep_best = loss_train, epoch
        path_saver = dir_saver + f'{prefix}_TrainLossMin_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.pkl'
        print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best} | Train_Loss_Best = {loss_train_best:.4f}')
        
        torch.save(model.eval().state_dict(), path_saver)
        
        plt.figure(figsize = (15, 10))
        plt.plot(range(epoch), loss_train_list, label = 'Train')
        if evaluation == 'True': plt.plot(range(epoch), loss_test_list, label = 'Test')
        plt.title(f'{prefix} CE Loss: layers={layers}|heads={heads}|embedsize={embed_dim}|batch_toks={batch_toks}')
        plt.legend()
        plt.savefig(f'./figures/{prefix}_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.tif')


# 在训练结束后，加载保存的最佳模型，并在训练数据和测试数据上评估其性能。将结果保存到 CSV 文件中。

model.load_state_dict(torch.load(f'./saved_models/{prefix}_TrainLossMin_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.pkl'))
loss_test, mlm_results = eval_step(model, data_loader)
mlm_results.to_csv(f'./MLM_results/{prefix}_TrainLossMin_MLM_traindata_results_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.csv', index = False)
    
    
if evaluation == 'True':
    model.load_state_dict(torch.load(f'./saved_models/{prefix}_TestLossMin_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.pkl'))
    loss_test, mlm_results = eval_step(model, test_dataloader)
    mlm_results.to_csv(f'./MLM_results/{prefix}_TestLossMin_MLM_testdata_results_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.csv', index = False)

    model.load_state_dict(torch.load(f'./saved_models/{prefix}_TrainLossMin_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.pkl'))
    loss_test, mlm_results = eval_step(model, test_dataloader)
    mlm_results.to_csv(f'./MLM_results/{prefix}_TrainLossMin_MLM_testdata_results_{layers}layers_{heads}heads_{embed_dim}embedsize_{batch_toks}batchToks.csv', index = False)

