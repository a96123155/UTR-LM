# netstat -ntlp | grep 3456
# nohup python -m torch.distributed.launch --nproc_per_node=3 --master_port 3456 Finetune_extract_append_predictor_CellLine_10fold-lr-huber-DDP.py --device_ids 0,1,2 --prefix DDP_M3.4 --cell_line Muscle --label_type te_log --seq_type utr --inp_len 100 --lr 0.1 --epochs 300 --huber_loss --modelfile /home/ubuntu/esm2/saved_models/ESM2SI_3.4_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl --finetune --bos_emb --train_n_atg > /home/ubuntu/esm2/Cao/outputs/DDP_M3.4.out 2>&1 &

# nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 3456 Finetune_extract_append_predictor_CellLine_10fold-lr-huber-DDP.py --device_ids 0,1,2,3 --prefix DDP_M3.4 --cell_line Muscle --label_type te_log --seq_type utr --inp_len 100 --huber_loss --modelfile /home/ubuntu/esm2/saved_models/ESM2SI_3.4_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl --finetune --bos_emb --train_n_atg --load_wholemodel --finetune_modeldir /home/ubuntu/esm2/Cao/saved_models/ --epochs 300 --lr 0.001 --log_interval 10 > /home/ubuntu/esm2/Cao/outputs/DDP_M3.4.out 2>&1 &

# {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
import os
import argparse
from argparse import Namespace
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import esm
from esm.data import *
from esm.model.esm2 import ESM2
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


import numpy as np
import pandas as pd
import random
import math
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from sklearn import preprocessing
from copy import deepcopy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='2,3', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
parser.add_argument('--log_interval', type=int, default=10, help="Log Interval")

parser.add_argument('--prefix', type=str, default = 'NSLDDP_M1')
parser.add_argument('--cell_line', type=str, default = 'Muscle')
parser.add_argument('--label_type', type=str, default = 'te_log')
parser.add_argument('--seq_type', type=str, default = 'utr')
parser.add_argument('--inp_len', type = int, default = 100)

parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--epochs', type = int, default = 300)
parser.add_argument('--cnn_layers', type = int, default = 0)
parser.add_argument('--nodes', type = int, default = 40)
parser.add_argument('--dropout3', type = float, default = 0.2)
parser.add_argument('--patience', type = int, default = 0)
parser.add_argument('--test1fold', action = 'store_true')
parser.add_argument('--huber_loss', action = 'store_true')

parser.add_argument('--load_wholemodel', action = 'store_true') ## if --finetune: False
# parser.add_argument('--init_epochs', type = int, default = 0)
parser.add_argument('--modelfile', type = str, default = '/home/ubuntu/esm2/saved_models/ESM2SI_3.4_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl')

parser.add_argument('--finetune_modeldir', type = str, default = '/home/ubuntu/esm2/Cao/saved_models')
# CVESM2lr1e-5_DDP9.1_unmod_1_10folds_rl_LabelScalerTrue_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_fold0_epoch19_lr0.1.pt
parser.add_argument('--finetune', action = 'store_true') ## if --finetune: False
parser.add_argument('--avg_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--bos_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--train_atg', action = 'store_true') ## if --finetune: False
parser.add_argument('--train_n_atg', action = 'store_true') ## if --finetune: False
parser.add_argument('--magic', action = 'store_true') ## if --finetune: False

args = parser.parse_args()
print(args)

global cell_line, seq_type, inp_len, layers, heads, embed_dim, batch_toks, inp_len, device_ids, device, train_obj_col, epoch

cell_line = args.cell_line
seq_type = args.seq_type
inp_len = args.inp_len
prefix = f'NSLCVESM2lr1e-5_{args.prefix}'

model_info = args.modelfile.split('/')[-1].split('_')
for item in model_info:
    if 'layers' in item: 
        layers = int(item[0])
    elif 'heads' in item:
        heads = int(item[:-5])
    elif 'embedsize' in item:
        embed_dim = int(item[:-9])
    elif 'batchToks' in item:
        batch_toks = 4096

device_ids = list(map(int, args.device_ids.split(',')))
dist.init_process_group(backend='nccl')
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
torch.cuda.set_device(device)
print(f'***********Device_ids = {device_ids} | device = {device}***********')

repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = True
return_representation = True

filename = f'{prefix}_{cell_line}_{args.label_type}_{seq_type}_seqlen{inp_len}_AvgEmb{args.avg_emb}_BosEmb{args.bos_emb}_CNNlayer{args.cnn_layers}_epoch{args.epochs}_patiences{args.patience}_nodes{args.nodes}_dropout3{args.dropout3}_finetune{args.finetune}_huberloss{args.huber_loss}_magic{args.magic}_lr{args.lr}'
print(filename)

    
class CNN_linear(nn.Module):
    def __init__(self, 
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        
        super(CNN_linear, self).__init__()
        
        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = args.nodes
        self.cnn_layers = args.cnn_layers
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = args.dropout3
        
        self.esm2 = ESM2(num_layers = layers,
                                 embed_dim = embed_dim,
                                 attention_heads = heads,
                                 alphabet = alphabet)
        
        self.conv1 = nn.Conv1d(in_channels = self.embedding_size, 
                      out_channels = self.nbr_filters, kernel_size = self.filter_len, padding = self.border_mode)
        self.conv2 = nn.Conv1d(in_channels = self.nbr_filters, 
                      out_channels = self.nbr_filters, kernel_size = self.filter_len, padding = self.border_mode)
        
        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        if args.avg_emb or args.bos_emb:
            self.fc = nn.Linear(in_features = embed_dim, out_features = self.nodes)
        else:
            self.fc = nn.Linear(in_features = inp_len * embed_dim, out_features = self.nodes)
        if args.avg_emb or args.bos_emb:
            self.linear = nn.Linear(in_features = self.nbr_filters, out_features = self.nodes)
        else:
            self.linear = nn.Linear(in_features = inp_len * self.nbr_filters, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 1)
        self.direct_output = nn.Linear(in_features = embed_dim, out_features = 1)
        self.magic_output = nn.Linear(in_features = 1, out_features = 1)
            
    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation = True):
        
        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        if args.avg_emb:
            x = x["representations"][layers][:, 1 : inp_len+1].mean(1)
            x_o = x.unsqueeze(2)
        elif args.bos_emb:
            x = x["representations"][layers][:, 0]
            x_o = x.unsqueeze(2)
        else:
            x_o = x["representations"][layers][:, 1 : inp_len+1]
            x_o = x_o.permute(0, 2, 1)

        if self.cnn_layers >= 1:
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2: 
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3: 
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)
        
#         if self.cnn_layers >= 1: 
        x = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x)
            else:
                o_linear = self.fc(x)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x)
#         print(o.shape)
        if args.magic:
            o = self.magic_output(o)
            
        if self.cnn_layers == -1:
            return o, (o.abs().sum().detach().cpu(), o.sum().detach().cpu())
        elif self.cnn_layers == 0:
            return o, (o.abs().sum().detach().cpu(), o.sum().detach().cpu())
        elif self.cnn_layers == 1:
            return o, (x_cnn1.abs().sum().detach().cpu(), x_cnn1.sum().detach().cpu(),\
                       x_o.sum().detach().cpu(),\
                       o_linear.abs().sum().detach().cpu(), o_linear.sum().detach().cpu(),\
                       o.abs().sum().detach().cpu(), o.sum().detach().cpu())
        elif self.cnn_layers == 2:
            return o, (x_cnn1.abs().sum().detach().cpu(), x_cnn1.sum().detach().cpu(),\
                       x_cnn2.abs().sum().detach().cpu(), x_cnn2.sum().detach().cpu(),\
                       x_relu2.sum().detach().cpu(),\
                       o_linear.abs().sum().detach().cpu(), o_linear.sum().detach().cpu(),\
                       o.abs().sum().detach().cpu(), o.sum().detach().cpu())
        elif self.cnn_layers == 3:
            return o, (x_cnn1.abs().sum().detach().cpu(), x_cnn1.sum().detach().cpu(),\
                       x_cnn2.abs().sum().detach().cpu(), x_cnn2.sum().detach().cpu(),\
                       x_cnn3.abs().sum().detach().cpu(), x_cnn3.sum().detach().cpu(),\
                       x_relu3.sum().detach().cpu(),\
                       o_linear.abs().sum().detach().cpu(), o_linear.sum().detach().cpu(),\
                       o.abs().sum().detach().cpu(), o.sum().detach().cpu())

def train_step(data, batches_loader, model):
    cnn_outputs_list, cnn_abs_ouputs_list = [], []
    relu_outputs_list = []
    linear_outputs_list, linear_abs_ouputs_list = [], []
    cnn_weight_list, cnn_abs_weight_list, cnn_grad_list, cnn_abs_grad_list = [], [], [], []
    linear_weight_list, linear_abs_weight_list, linear_grad_list, linear_abs_grad_list = [], [], [], []
        
    model.train()
    y_pred_list, y_true_list, loss_list = [], [], []
    
    for i, batch in tqdm(enumerate(train_batches_loader)):
        batch = np.array(torch.LongTensor(batch))
        e_data = data.iloc[batch]
        
        dataset = FastaBatchedDataset(e_data.loc[:, args.label_type], e_data.utr, mask_prob = 0.0)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                collate_fn=alphabet.get_batch_converter(), 
                                                batch_size=len(batch), 
                                                shuffle = False)
   
        for (labels, strs, masked_strs, toks, masked_toks, _) in dataloader:
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)

            outputs, _ = model(toks, return_representation = True, return_contacts=True)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.cpu().detach())

            y_true_list.extend(labels.cpu().reshape(-1).tolist())

            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            y_pred_list.extend(y_pred)
#             break
#         break
    loss_epoch = float(torch.Tensor(loss_list).mean())
    print(f'Train: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end = '')
#     print(f'max(y_true_list) = {max(y_true_list):.4f}, max(y_pred_list) = {max(y_pred_list):.4f}')
#     print(f'min(y_true_list) = {min(y_true_list):.4f}, min(y_pred_list) = {min(y_pred_list):.4f}')
    
    metrics = performances(y_true_list, y_pred_list)
    return metrics, loss_epoch, (
    loss_list, cnn_outputs_list, cnn_abs_ouputs_list, relu_outputs_list, linear_outputs_list, linear_abs_ouputs_list, cnn_weight_list, cnn_abs_weight_list, cnn_grad_list, cnn_abs_grad_list, linear_weight_list, linear_abs_weight_list, linear_grad_list, linear_abs_grad_list)


def eval_step(dataloader, model, epoch, data = None):
    model.eval()
    y_pred_list, y_true_list, loss_list = [], [], []
    strs_list = []
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(dataloader):
            strs_list.extend(strs)
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
            
            outputs, _ = model(toks, return_representation = True, return_contacts=True) # scaled_log2

            y_true_list.extend(labels.cpu().reshape(-1).tolist())

            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            y_pred_list.extend(y_pred)
        
        loss_epoch = criterion(torch.Tensor(y_pred_list).reshape(-1,1), torch.Tensor(y_true_list).reshape(-1,1))
        
        print(f'Test: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end = '')
        metrics = performances(y_true_list, y_pred_list)
        print(f'max(y_true_list) = {max(y_true_list):.4f}, max(y_pred_list) = {max(y_pred_list):.4f}')
        print(f'min(y_true_list) = {min(y_true_list):.4f}, min(y_pred_list) = {min(y_pred_list):.4f}')
        e_pred = pd.DataFrame([strs_list, y_true_list, y_pred_list], index = ['train_seqs', 'y_true', 'y_pred']).T
        
        if data is not None: 
            data_pred = pd.concat([data, e_pred], axis = 1)
        else:
            data_pred = e_pred
    return metrics, loss_epoch, e_pred

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def performances(label, pred):
    label, pred = list(label), list(pred)
    try:
        r = r2(label, pred)
    except:
        r = -1
    try:
        pearson_r = pearsonr(label, pred)[0]
    except:
        pearson_r = -1
    try:
        sp_cor = spearmanr(label, pred)[0]
    except:
        sp_cor = -1
    print(f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f}')
        
    return [r, pearson_r, sp_cor]

def generate_dataset_dataloader(e_data, obj_col):
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data.utr, mask_prob = 0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader


def generate_trainbatch_loader(e_data, obj_col):
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data.utr, mask_prob = 0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    batches_sampler = DistributedSampler(batches, shuffle = True)
    batches_loader = torch.utils.data.DataLoader(batches, 
                                                 batch_size = 1,
                                                 num_workers = 8,
                                                 sampler = batches_sampler)
    print(f"{len(dataset)} sequences")
    print(f'{len(batches)} batches')
    #print(f' Batches: {batches[0]}')
    return dataset, batches, batches_sampler, batches_loader

def ATG_pred_performance(data_pred):
    
    data_atg_pred = data_pred[data_pred.train_seqs.str.contains('ATG')]
    data_n_atg_pred = data_pred[~data_pred.train_seqs.str.contains('ATG')]
    
    if len(data_atg_pred) != 0:
        metrics_data_atg = performances(data_atg_pred['y_true'], data_atg_pred['y_pred'])
    else:
        metrics_data_atg = [np.nan, np.nan, np.nan]
        
    if len(data_n_atg_pred) != 0:
        metrics_data_n_atg = performances(data_n_atg_pred['y_true'], data_n_atg_pred['y_pred'])
    else:
        metrics_data_n_atg = [np.nan, np.nan, np.nan]
        
    return data_atg_pred, data_n_atg_pred, metrics_data_atg, metrics_data_n_atg
#######

alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

print('====Load Data====')
all_data = pd.read_csv(f'/home/ubuntu/5UTR_Optimizer-master/data/{cell_line}_sequence.csv')
if args.train_atg: 
    data = all_data[all_data[seq_type].str.contains('ATG')]
if args.train_n_atg: 
    data = all_data[~all_data[seq_type].str.contains('ATG')]
else:
    data = deepcopy(all_data)

KF = KFold(n_splits=10, shuffle=True)
fig, axes = plt.subplots(nrows = 10, ncols = 7, figsize = (25, 40))

loss_train_repeat_dict, loss_val_repeat_dict, loss_alldata_repeat_dict = dict(), dict(), dict()
metrics_repeat_dict = dict()
best_epoch_list = []
for i, (train_index, val_index) in enumerate(KF.split(data)):

    print(f'====Begin Train Fold = {i}====')
    
    e_train = data.iloc[train_index, :]
    e_val = data.iloc[val_index, :]
        
    train_dataset, train_batches, train_batches_sampler, train_batches_loader = generate_trainbatch_loader(e_train, args.label_type)
    val_dataset, val_dataloader = generate_dataset_dataloader(e_val, args.label_type)
    
    model = CNN_linear().to(device)
    print(model)
    storage_id = int(device_ids[args.local_rank])
    if args.load_wholemodel: 
        for f in os.listdir(args.finetune_modeldir):
            if prefix in f and f'fold{i}' in f:
                finetune_modelfile = f'{args.finetune_modeldir}/{f}'
                model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(finetune_modelfile, map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
                init_epochs = int(finetune_modelfile.split('_')[-1][5:-3])#19
                print(f'--Finetune File : {finetune_modelfile}')
                break
    else:
        init_epochs = 0
        model.esm2.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(args.modelfile, map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
    if i == 0: model = DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]], output_device=device_ids[args.local_rank], find_unused_parameters=True)

    if not args.finetune:
        for name, value in model.named_parameters():
            if 'esm2' in name:
                value.requires_grad = False
            print(name, value.requires_grad)
    if args.magic:
        for name, value in model.named_parameters():
            if 'magic_output' not in name:
                value.requires_grad = False
            print(name, value.requires_grad)
            
    params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = torch.optim.Adam(params,
#                                  lr = args.lr, 
#                                  betas = (0.9, 0.999),
#                                  eps = 1e-08)
    optimizer = torch.optim.SGD(params,
                                 lr = args.lr, 
                                 momentum=0.9,
                                 weight_decay = 1e-4)

    loss_best, ep_best, r2_best = np.inf, -1, -1
    loss_train_list, loss_val_list = [], []
    for epoch in trange(init_epochs+1, init_epochs + args.epochs + 1):
        train_batches_sampler.set_epoch(epoch)
        
        if args.huber_loss:
            criterion = torch.nn.HuberLoss()
        else:
            criterion = torch.nn.MSELoss()
        metrics_train, loss_train, _ = train_step(e_train, train_batches_loader, model)
        loss_train_list.append(loss_train)
    
        if args.local_rank == 0:
            metrics_val, loss_val, _ = eval_step(val_dataloader, model, epoch)
            loss_val_list.append(loss_val)
            if args.epochs >= args.patience:
                if metrics_val[0] > r2_best: 
                    path_saver = f'/home/ubuntu/esm2/Cao/saved_models/{filename}_fold{i}_epoch{epoch}.pt'
                    r2_best, ep_best = metrics_val[0], epoch
                    
                    
                    torch.save(model.eval().state_dict(), path_saver) # 
                    print(f'****Saving model in {path_saver}: Best epoch = {ep_best} | Train Loss = {loss_train:.4f} |  Val Loss = {loss_val:.4f} | R2_best = {r2_best:.4f}')
                    model_best = deepcopy(model)
                    
        ##### results
    best_epoch_list.append(ep_best)
    if args.local_rank == 0:
        print('=====Generate results=====')

        alldata_dataset, alldata_dataloader = generate_dataset_dataloader(all_data, args.label_type)
        train_dataset, train_dataloader = generate_dataset_dataloader(e_train, args.label_type)

        metrics_alldata, loss_alldata, alldata_pred = eval_step(alldata_dataloader, model_best, ep_best)
        metrics_train, loss_train, e_train_pred = eval_step(train_dataloader, model_best, ep_best)
        metrics_val, loss_val, e_val_pred = eval_step(val_dataloader, model_best, ep_best)

        print('====ATG Analysis====')
        alldata_atg_pred, alldata_n_atg_pred, metrics_alldata_atg, metrics_alldata_n_atg = ATG_pred_performance(alldata_pred)

        e_train_atg_pred, e_train_n_atg_pred, metrics_train_atg, metrics_train_n_atg = ATG_pred_performance(e_train_pred)
        e_val_atg_pred, e_val_n_atg_pred, metrics_val_atg, metrics_val_n_atg = ATG_pred_performance(e_val_pred)

        print('====Save y_pred====')
        alldata_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_alldata_fold{i}.csv', index = False)
        e_train_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_train_fold{i}.csv', index = False)
        e_val_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_val_fold{i}.csv', index = False)

        alldata_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_alldata_atg_fold{i}.csv', index = False)
        alldata_n_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_alldata_noatg_fold{i}.csv', index = False)

        e_train_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_train_atg_fold{i}.csv', index = False)
        e_train_n_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_train_noatg_fold{i}.csv', index = False)

        e_val_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_val_atg_fold{i}.csv', index = False)
        e_val_n_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_val_noatg_fold{i}.csv', index = False)

        print('====Save Metrics====')
        metrics_repeat_dict[i] = metrics_alldata + metrics_train + metrics_val + metrics_alldata_atg + metrics_train_atg + metrics_val_atg + metrics_alldata_n_atg + metrics_train_n_atg + metrics_val_n_atg

        metrics_repeat_df = pd.DataFrame(metrics_repeat_dict, 
                                        index = ['All_R2', 'All_PearsonR', 'All_SpearmanR',
                                                 'Train_R2', 'Train_PearsonR', 'Train_SpearmanR',
                                                'Test_R2', 'Test_PearsonR', 'Test_SpearmanR',
                                                'All_R2_ATG', 'All_PearsonR_ATG', 'All_SpearmanR_ATG',
                                                'Train_R2_ATG', 'Train_PearsonR_ATG', 'Train_SpearmanR_ATG',
                                                'Test_R2_ATG', 'Test_PearsonR_ATG', 'Test_SpearmanR_ATG',
                                                'All_R2_noATG', 'All_PearsonR_noATG', 'All_SpearmanR_noATG',
                                                'Train_R2_noATG', 'Train_PearsonR_noATG', 'Train_SpearmanR_noATG',
                                                'Test_R2_noATG', 'Test_PearsonR_noATG', 'Test_SpearmanR_noATG'
                                                ]).T

        metrics_repeat_df['best_epoch'] = best_epoch_list
        metrics_repeat_df.fillna(-1, inplace = True)
        metrics_repeat_df.loc['mean'] = metrics_repeat_df[metrics_repeat_df['Train_SpearmanR'] != -1].mean(axis = 0)
        metrics_repeat_df.loc['std'] = metrics_repeat_df[metrics_repeat_df['Train_SpearmanR'] != -1].std(axis = 0)
        metrics_repeat_df.to_csv(f'/home/ubuntu/esm2/Cao/results/{filename}_metrics.csv', index = True)

        loss_train_repeat_dict[i], loss_val_repeat_dict[i], loss_alldata_repeat_dict[i] = loss_train_list, loss_val_list, [loss_alldata]*len(loss_train_list)

        ##### Figures

        axes[i, 0].hist(e_train_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
        axes[i, 0].hist(e_train_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
        axes[i, 0].set_title(f'Fold{i} Train|n={len(e_train_pred)}|R2={metrics_train[0]:.4f}', fontsize = 12)
        axes[i, 0].legend(fontsize = 12)

        axes[i, 1].hist(e_train_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
        axes[i, 1].hist(e_train_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
        axes[i, 1].set_title(f'Fold{i} Train ATG|n={len(e_train_atg_pred)}|R2={metrics_train_atg[0]:.4f}', fontsize = 12)
        axes[i, 1].legend(fontsize = 12)

        axes[i, 2].hist(e_train_n_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
        axes[i, 2].hist(e_train_n_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
        axes[i, 2].set_title(f'Fold{i} Train No-ATG|n={len(e_train_n_atg_pred)}|R2={metrics_train_n_atg[0]:.4f}', fontsize = 12)
        axes[i, 2].legend(fontsize = 12)

        axes[i, 3].hist(e_val_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
        axes[i, 3].hist(e_val_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
        axes[i, 3].set_title(f'Fold{i} Val|n={len(e_val_pred)}|R2={metrics_val[0]:.4f}', fontsize = 12)
        axes[i, 3].legend(fontsize = 12)

        axes[i, 4].hist(e_val_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
        axes[i, 4].hist(e_val_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
        axes[i, 4].set_title(f'Fold{i} Val ATG|n={len(e_val_atg_pred)}|R2={metrics_val_atg[0]:.4f}', fontsize = 12)
        axes[i, 4].legend(fontsize = 12)

        axes[i, 5].hist(e_val_n_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
        axes[i, 5].hist(e_val_n_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
        axes[i, 5].set_title(f'Fold{i} Val No-ATG|n={len(e_val_n_atg_pred)}|R2={metrics_val_n_atg[0]:.4f}', fontsize = 12)
        axes[i, 5].legend(fontsize = 12)


        axes[i, 6].plot(range(init_epochs, epoch), loss_train_list, label = f'Train_{i}={loss_train:.4f}(Best={ep_best})', linestyle = ':')
        axes[i, 6].plot(range(init_epochs, epoch), loss_val_list, label = f'Val_{i}={loss_val:.4f}(Best={ep_best})')
        axes[i, 6].plot(range(init_epochs, epoch), loss_alldata_repeat_dict[i], label = f'All_{i}={loss_alldata:.4f}(Best={ep_best})', linestyle = '--')
        axes[i, 6].legend(fontsize = 12)

        plt.savefig(f'/home/ubuntu/esm2/Cao/figures/{filename}.tif')
    if args.test1fold: break
        
print(f'/home/ubuntu/esm2/Cao/saved_models/{filename}_fold{i}.pkl')    
print(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_train_fold{i}.csv')
print(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_test_fold{i}.csv')
print(f'/home/ubuntu/esm2/Cao/results/{filename}_metrics.csv')    
print(f'/home/ubuntu/esm2/Cao/figures/{filename}.tif')