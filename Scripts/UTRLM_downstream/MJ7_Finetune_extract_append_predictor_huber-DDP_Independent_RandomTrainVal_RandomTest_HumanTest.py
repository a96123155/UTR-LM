# CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 5001 MJ3_Finetune_extract_append_predictor_Sample_10fold-lr-huber-DDP.py --device_ids 0,1,2,3 --label_type rl --epochs 1 --huber_loss --train_file 4.1_train_data_GSM3130435_egfp_unmod_1.csv --prefix ESM2SISS_FS4.1.ep93.1e-2.dr5 --lr 1e-2 --dropout3 0.5 --modelfile /home/yanyichu/UTR-LM/Model/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl --finetune --bos_emb --test1fold

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
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


import numpy as np
import pandas as pd
import random
import math
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
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

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0,1,2', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
parser.add_argument('--log_interval', type=int, default=50, help="Log Interval")
parser.add_argument('--seed', type = int, default = 1337)

parser.add_argument('--prefix', type=str, default = 'BenchmarkSplit')
parser.add_argument('--label_type', type=str, default = 'rl')
parser.add_argument('--seq_type', type=str, default = 'utr_100')
parser.add_argument('--inp_len', type=int, default = 100)

parser.add_argument('--epochs', type = int, default = 300)
parser.add_argument('--cnn_layers', type = int, default = 0)
parser.add_argument('--nodes', type = int, default = 40)
parser.add_argument('--dropout3', type = float, default = 0.5)
parser.add_argument('--lr', type = float, default = 1e-2)
parser.add_argument('--folds', type = int, default = 10)
parser.add_argument('--patience', type = int, default = 0)
parser.add_argument('--test1fold', action = 'store_true')
parser.add_argument('--huber_loss', action = 'store_true')

parser.add_argument('--train_file', type = str, default = 'VaryLengthRandomTrain_sequence.csv')

parser.add_argument('--load_wholemodel', action = 'store_true') ## if --finetune: False
parser.add_argument('--init_epochs', type = int, default = 0)
parser.add_argument('--modelfile', type = str, default = '/home/yanyichu/UTR-LM/saved_models/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl')

parser.add_argument('--finetune_modelfile', type = str, default = '/home/yanyichu/UTR-LM/saved_models/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl')

parser.add_argument('--finetune', action = 'store_true') ## if --finetune: False
parser.add_argument('--scaler', action = 'store_true') ## if --finetune: False
parser.add_argument('--log2', action = 'store_true') ## if --finetune: False
parser.add_argument('--avg_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--bos_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--train_atg', action = 'store_true') ## if --finetune: False
parser.add_argument('--train_n_atg', action = 'store_true') ## if --finetune: False
parser.add_argument('--magic', action = 'store_true') ## if --finetune: False

args = parser.parse_args()
print(args)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

cell_line = args.train_file.replace('.csv', '')
        

global layers, heads, embed_dim, batch_toks, inp_len, device_ids, device, train_obj_col, epoch
model_info = args.modelfile.split('/')[-1].split('_')
for item in model_info:
    if 'layers' in item: 
        layers = int(item[0])
    elif 'heads' in item:
        heads = int(item[:-5])
    elif 'embedsize' in item:
        embed_dim = int(item[:-9])
    elif 'batchToks' in item:
        print(item)
        batch_toks = 4096
inp_len = args.inp_len
prefix = f'MJ7_seed{seed}_{args.prefix}'
    
device_ids = list(map(int, args.device_ids.split(',')))
dist.init_process_group(backend='nccl')
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
torch.cuda.set_device(device)

repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = True
return_representation = True

filename = f'{prefix}_{cell_line}_{args.seq_type}_{args.folds}folds_{args.label_type}_LabelScaler{args.scaler}_LabelLog2{args.log2}_AvgEmb{args.avg_emb}_BosEmb{args.bos_emb}_CNNlayer{args.cnn_layers}_epoch{args.epochs}_nodes{args.nodes}_dropout3{args.dropout3}_finetune{args.finetune}_huberloss{args.huber_loss}_lr{args.lr}'
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
        
        if 'SISS' in args.modelfile:
            self.esm2 = ESM2_SISS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        elif 'SS' in args.modelfile:
            self.esm2 = ESM2_SS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        else:
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
        if self.cnn_layers == -1: self.direct_output = nn.Linear(in_features = embed_dim, out_features = 1)
        if args.magic: self.magic_output = nn.Linear(in_features = 1, out_features = 1)
            
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
        return o
    
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
        
        dataset = FastaBatchedDataset(e_data.loc[:, train_obj_col], e_data.utr, mask_prob = 0.0)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                collate_fn=alphabet.get_batch_converter(), 
                                                batch_size=len(batch), 
                                                shuffle = False)
   
        for (labels, strs, masked_strs, toks, masked_toks, _) in dataloader:
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)

            outputs = model(toks, return_representation = True, return_contacts=True)
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
    return metrics, loss_epoch


def eval_step(dataloader, model, epoch, data = None):
    model.eval()
    y_pred_list, y_true_list, loss_list = [], [], []
    strs_list = []
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(dataloader):
            strs_list.extend(strs)
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
            
            outputs = model(toks, return_representation = True, return_contacts=True) # scaled_log2

            y_true_list.extend(labels.cpu().reshape(-1).tolist())

            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            y_pred_list.extend(y_pred)
        
        if args.scaler:
            scaler = preprocessing.StandardScaler()
            scaler.fit(np.array(y_true_list).reshape(-1,1))
            y_pred_list = scaler.inverse_transform(np.array(y_pred_list).reshape(-1,1)).reshape(-1) #Unscaled to RL
        if args.log2:
            y_pred_list = list(map(lambda x:math.pow(2,x), y_pred_list))
        loss_epoch = criterion(torch.Tensor(y_pred_list).reshape(-1,1), torch.Tensor(y_true_list).reshape(-1,1))
        
        print(f'Test: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end = '')
        metrics = performances(y_true_list, y_pred_list)
        print(f'max(y_true_list) = {max(y_true_list):.4f}, max(y_pred_list) = {max(y_pred_list):.4f}')
        print(f'min(y_true_list) = {min(y_true_list):.4f}, min(y_pred_list) = {min(y_pred_list):.4f}')
        e_pred = pd.DataFrame([strs_list, y_true_list, y_pred_list], index = ['utr', 'y_true', 'y_pred']).T
        
        if data is not None: 
            data_pred = pd.merge(e_pred, data, on = ['utr'])
        else:
            data_pred = e_pred
    return metrics, loss_epoch, data_pred

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def performances(label, pred):
    label, pred = list(label), list(pred)
    
    r = r2(label, pred)
    R2 = r2_score(label, pred)
    
    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)
    
    try:
        pearson_r = pearsonr(label, pred)[0]
    except:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(label, pred)[0]
    except:
        sp_cor = -1e-9
    
    print(f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f} | R-squared = {R2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
        
    return [r, pearson_r, sp_cor, R2, rmse, mae]

def generate_dataset_dataloader(e_data, obj_col):
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data[args.seq_type], mask_prob = 0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader


def generate_trainbatch_loader(e_data, obj_col):
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data[args.seq_type], mask_prob = 0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    batches_sampler = DistributedSampler(batches, shuffle = True)
    batches_loader = torch.utils.data.DataLoader(batches, 
                                                 batch_size = 1,
                                                 num_workers = 1,
                                                 sampler = batches_sampler)
    print(f"{len(dataset)} sequences")
    print(f'{len(batches)} batches')
    #print(f' Batches: {batches[0]}')
    return dataset, batches, batches_sampler, batches_loader
#######

alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

print('====Load Data====')
train_data = pd.read_csv(f'/home/yanyichu/UTR-LM/Data/IndependentTest_VaryLength_Sample/{args.train_file}')
test_random_data = pd.read_csv(f'/home/yanyichu/UTR-LM/Data/IndependentTest_VaryLength_Sample/VaryLengthRandomTest_sequence_num7600.csv')
test_human_data = pd.read_csv(f'/home/yanyichu/UTR-LM/Data/IndependentTest_VaryLength_Sample/VaryLengthHumanTest_sequence_num7600.csv')

data = deepcopy(train_data)
data = data.sample(frac=1, random_state = seed).reset_index(drop=True)

print(train_data.shape, test_random_data.shape, test_human_data.shape)

KF = KFold(n_splits=args.folds, shuffle=False)

loss_train_repeat_dict, loss_val_repeat_dict, loss_test_repeat_dict = dict(), dict(), dict()
metrics_repeat_dict = dict()
best_epoch_list = []
traincv_data_pred_df = pd.DataFrame()
for i, (train_index, val_index) in enumerate(KF.split(data)):
    print(f'====Begin Train Fold = {i}====')
    e_train = data.iloc[train_index, :]
    e_val = data.iloc[val_index, :]
    print('-----', data.shape, e_train.shape, e_val.shape)

    if args.scaler and args.log2:
        train_obj_col = f'{args.label_type}_log2_scaled'
        e_train.loc[:, train_obj_col] = preprocessing.StandardScaler().fit_transform(e_train.loc[:,f'{args.label_type}_log2'].values.reshape(-1,1))
    elif args.scaler and not args.log2:
        train_obj_col = f'{args.label_type}_scaled'
        e_train.loc[:, train_obj_col] = preprocessing.StandardScaler().fit_transform(e_train.loc[:,f'{args.label_type}'].values.reshape(-1,1))
    elif not args.scaler and args.log2:
        train_obj_col = f'{args.label_type}_log2'
    else:
        train_obj_col = args.label_type
    
    train_dataset, train_batches, train_batches_sampler, train_batches_loader = generate_trainbatch_loader(e_train, train_obj_col)
    val_dataset, val_dataloader = generate_dataset_dataloader(e_val, args.label_type)
    
    model = CNN_linear().to(device)
    storage_id = int(device_ids[args.local_rank])
    if args.load_wholemodel: 
        model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(args.finetune_modelfile, map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
    else:
        print(f'********Device IDs = {device_ids}, cuda:{device_ids[args.local_rank]}')
        model.esm2.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(args.modelfile, map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
    if i == 0: model = DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]], output_device=device_ids[args.local_rank], find_unused_parameters=True)

    if not args.finetune:
        for name, value in model.named_parameters():
            if 'esm2' in name:
                value.requires_grad = False
            print(name, value.requires_grad)
    else:
        for name, value in model.named_parameters():
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
#     hidden_visualization_dict = dict()
    for epoch in trange(args.init_epochs+1, args.init_epochs + args.epochs + 2):
        train_batches_sampler.set_epoch(epoch)
 
        if args.huber_loss:
            criterion = torch.nn.HuberLoss()
        else:
            criterion = torch.nn.MSELoss()
        metrics_train, loss_train = train_step(e_train, train_batches_loader, model)
#         hidden_visualization_dict[epoch] = hidden_visualization
        loss_train_list.append(loss_train)
    
        if epoch == args.init_epochs+1:
            model_best = deepcopy(model)
        if args.local_rank == 0:
            metrics_val, loss_val, _ = eval_step(val_dataloader, model, epoch)
            loss_val_list.append(loss_val)
            if args.epochs >= args.patience:
                if metrics_val[2] > r2_best: 
                    path_saver = f'/home/yanyichu/UTR-LM/Sample/saved_models/{filename}_fold{i}_epoch{epoch}.pt'
                    r2_best, ep_best = metrics_val[2], epoch
                    
                    torch.save(model.eval().state_dict(), path_saver) # 
                    print(f'****Saving model in {path_saver}: Best epoch = {ep_best} | Train Loss = {loss_train:.4f} |  Val Loss = {loss_val:.4f} | SpearmanR_best = {r2_best:.4f}')
                    model_best = deepcopy(model)
                    
        ##### results

            if epoch % args.log_interval == 0:
                print('=====Generate results=====')
                
                train_dataset, train_dataloader = generate_dataset_dataloader(e_train, args.label_type)
                val_dataset, val_dataloader = generate_dataset_dataloader(e_val, args.label_type)
                test_random_dataset, test_random_dataloader = generate_dataset_dataloader(test_random_data, args.label_type)
                test_human_dataset, test_human_dataloader = generate_dataset_dataloader(test_human_data, args.label_type)

        #         model_best = CNN_linear().to(device)
        #         model_best.load_state_dict(torch.load(path_saver), load_state_dict(strict=False))
        #         model_best = torch.load(path_saver)#, map_location = lambda storage, loc : storage.cuda(0)))

                metrics_train, loss_train, e_train_pred = eval_step(train_dataloader, model_best, ep_best, e_train)
                metrics_val, loss_val, e_val_pred = eval_step(val_dataloader, model_best, ep_best, e_val)
                metrics_random, loss_random, e_random_pred = eval_step(test_random_dataloader, model_best, ep_best, test_random_data)
                metrics_human, loss_human, e_human_pred = eval_step(test_human_dataloader, model_best, ep_best, test_human_data)

                
                print('====Save y_pred====')
                e_train_pred.to_csv(f'/home/yanyichu/UTR-LM/Sample/y_pred/{filename}_train_fold{i}.csv', index = False)
                e_val_pred.to_csv(f'/home/yanyichu/UTR-LM/Sample/y_pred/{filename}_val_fold{i}.csv', index = False)
                e_random_pred.to_csv(f'/home/yanyichu/UTR-LM/Sample/y_pred/{filename}_testrandom_fold{i}.csv', index = False)
                e_human_pred.to_csv(f'/home/yanyichu/UTR-LM/Sample/y_pred/{filename}_testhuman_fold{i}.csv', index = False)


                print('====Save Metrics====')
                metrics_repeat_dict[i] = metrics_random + metrics_human + metrics_train + metrics_val
                metrics_repeat_df = pd.DataFrame(metrics_repeat_dict, 
                                                index = ['Test_Random_r2', 'Test_Random_PearsonR', 'Test_Random_SpearmanR', 'Test_Random_R2', 'Test_Random_RMSE', 'Test_Random_MAE',
                                                        'Test_Human_r2', 'Test_Human_PearsonR', 'Test_Human_SpearmanR', 'Test_Human_R2', 'Test_Human_RMSE', 'Test_Human_MAE',
                                                        'Train_r2', 'Train_PearsonR', 'Train_SpearmanR', 'Train_R2', 'Train_RMSE', 'Train_MAE',
                                                        'val_r2', 'val_PearsonR', 'val_SpearmanR', 'Val_R2', 'Val_RMSE', 'Val_MAE'
                                                        ]).T
                
                metrics_repeat_df['best_epoch'] = ep_best
                metrics_repeat_df.fillna(-1, inplace = True)
                metrics_repeat_df.loc['mean'] = metrics_repeat_df[metrics_repeat_df['Train_SpearmanR'] != -1].mean(axis = 0)
                metrics_repeat_df.loc['std'] = metrics_repeat_df[metrics_repeat_df['Train_SpearmanR'] != -1].std(axis = 0)
                metrics_repeat_df.to_csv(f'/home/yanyichu/UTR-LM/Sample/results/{filename}_metrics.csv', index = True)

                loss_train_repeat_dict[i], loss_val_repeat_dict[i], loss_test_repeat_dict[i] = loss_train_list, loss_val_list, [loss_test]*len(loss_train_list)#epoch

                ##### Figures

                fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 10))

                axes[0, 0].hist(e_train_pred[args.label_type], bins = 100, alpha = 0.3, label = f'Y_True')
                axes[0, 0].hist(e_train_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
                axes[0, 0].set_title(f'All Train Data | {args.label_type} | n = {len(e_train_pred)} | R2 = {metrics_train[0]:.4f}', fontsize = 12)
                axes[0, 0].legend(fontsize = 12)

                axes[0, 1].hist(e_val_pred[args.label_type], bins = 100, alpha = 0.3, label = f'Y_True')
                axes[0, 1].hist(e_val_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
                axes[0, 1].set_title(f'All val Data | {args.label_type} | n = {len(e_val_pred)} | R2 = {metrics_val[0]:.4f}', fontsize = 12)
                axes[0, 1].legend(fontsize = 12)
                
                axes[0, 2].plot(range(args.init_epochs, epoch), loss_train_repeat_dict[i], label = f'MSELoss: Fold_Train_{i} = {loss_train:.4f} (Best = {ep_best})', linestyle = ':')
                axes[0, 2].plot(range(args.init_epochs, epoch), loss_val_repeat_dict[i], label = f'MSELoss: All_val_{i} = {loss_val:.4f} (Best = {ep_best})', linestyle = '--')
                axes[0, 2].legend(fontsize = 12)
                

                axes[1, 0].hist(e_random_pred[args.label_type], bins = 100, alpha = 0.3, label = f'Y_True')
                axes[1, 0].hist(e_random_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
                axes[1, 0].set_title(f'All Train ATG Data | {args.label_type} | n = {len(e_random_pred)} | R2 = {metrics_random[0]:.4f}', fontsize = 12)
                axes[1, 0].legend(fontsize = 12)

                axes[1, 1].hist(e_human_pred[args.label_type], bins = 100, alpha = 0.3, label = f'Y_True')
                axes[1, 1].hist(e_human_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
                axes[1, 1].set_title(f'All val ATG Data | {args.label_type} | n = {len(e_human_pred)} | R2 = {metrics_human[0]:.4f}', fontsize = 12)
                axes[1, 1].legend(fontsize = 12)

                plt.savefig(f'/home/yanyichu/UTR-LM/Sample/figures/{filename}_fold{i}.tif')
    if args.test1fold: break
    best_epoch_list.append(ep_best)
print(f'/home/yanyichu/UTR-LM/Sample/saved_models/{filename}_fold{i}.pkl')    
print(f'/home/yanyichu/UTR-LM/Sample/y_pred/{filename}_train_fold{i}.csv')
print(f'/home/yanyichu/UTR-LM/Sample/y_pred/{filename}_test_fold{i}.csv')
print(f'/home/yanyichu/UTR-LM/Sample/results/{filename}_metrics.csv')    
print(f'/home/yanyichu/UTR-LM/Sample/figures/{filename}_fold{i}.tif')
