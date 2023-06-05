# python Finetune_extract_append_predictor_CellLine_10fold-lr-huber.py --cell_line Muscle --prefix 11M.4.1.1.2 --epochs 2 --folds 10 --modelfile /home/ubuntu/esm2/saved_models/ESM2_1.5_five_species_TrainLossMin_6layers_16heads_256embedsize_4096batchToks.pkl --cnn_layers 0 --avg_emb --patience 0  --huber_loss --finetune --label_type te_log

# nohup python Finetune_extract_append_predictor_CellLine_10fold-lr-huber.py --cell_line Muscle --prefix 11M.4.1.1.2 --epochs 150 --folds 10 --modelfile /home/ubuntu/esm2/saved_models/ESM2_1.5_five_species_TrainLossMin_6layers_16heads_256embedsize_4096batchToks.pkl --cnn_layers 0 --avg_emb --patience 50  --huber_loss --finetune --label_type te_log > /home/ubuntu/esm2/Cao/outputs/11M.4.1.1.2.patience100.out 2>&1 &


# nohup python Finetune_extract_append_predictor_CellLine_10fold-lr-huber.py --cell_line pc3 --prefix 11P.4.1.1.2 --epochs 150 --folds 10 --modelfile /home/ubuntu/esm2/saved_models/ESM2_1.5_five_species_TrainLossMin_6layers_16heads_256embedsize_4096batchToks.pkl --cnn_layers 0 --avg_emb --patience 50  --huber_loss --finetune --label_type te_log > /home/ubuntu/esm2/Cao/outputs/11P.4.1.1.2.patience100.out 2>&1 &


# nohup python Finetune_extract_append_predictor_CellLine_10fold-lr-huber.py --cell_line HEK --prefix 11H.4.1.1.2 --epochs 150 --folds 10 --modelfile /home/ubuntu/esm2/saved_models/ESM2_1.5_five_species_TrainLossMin_6layers_16heads_256embedsize_4096batchToks.pkl --cnn_layers 0 --avg_emb --patience 50  --huber_loss --finetune --label_type te_log > /home/ubuntu/esm2/Cao/outputs/11H.4.1.1.2.patience100.out 2>&1 &

# nohup python Finetune_extract_append_predictor_CellLine_10fold-lr-huber.py --cell_line MuscleLR100 --prefix 11M.LR100.4.1.1.2 --epochs 150 --folds 10 --modelfile /home/ubuntu/esm2/saved_models/ESM2_1.5_five_species_TrainLossMin_6layers_16heads_256embedsize_4096batchToks.pkl --cnn_layers 0 --avg_emb --patience 50  --huber_loss --finetune --label_type te_median --seq_type utr_lr100 --inp_len 200 > /home/ubuntu/esm2/Cao/outputs/11M.LR100.1.1.2.patience100.out 2>&1 &


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

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default = '11M')
parser.add_argument('--cell_line', type=str, default = 'Muscle')
parser.add_argument('--label_type', type=str, default = 'te_log')
parser.add_argument('--seq_type', type=str, default = 'utr')

parser.add_argument('--inp_len', type = int, default = 100)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--cnn_layers', type = int, default = -1)
parser.add_argument('--nodes', type = int, default = 40)
parser.add_argument('--dropout3', type = float, default = 0.2)
parser.add_argument('--folds', type = int, default = 10)
parser.add_argument('--patience', type = int, default = 0)
parser.add_argument('--test1fold', action = 'store_true')
parser.add_argument('--huber_loss', action = 'store_true')

parser.add_argument('--gpu', type = str, default = '0')

parser.add_argument('--modelfile', type = str, default = '/home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')
parser.add_argument('--finetune', action = 'store_true') ## if --finetune: False
parser.add_argument('--avg_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--bos_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--train_atg', action = 'store_true') ## if --finetune: False
parser.add_argument('--train_n_atg', action = 'store_true') ## if --finetune: False

args = parser.parse_args()
print(args)

cell_line = args.cell_line
seq_type = args.seq_type
global layers, heads, embed_dim, batch_toks, inp_len
model_info = args.modelfile.split('/')[-1].split('_')
layers = int(model_info[5][:-6])
heads = int(model_info[6][:-5])
embed_dim = int(model_info[7][:-9])
batch_toks = int(model_info[8][:-13])
inp_len = args.inp_len
prefix = f'CVESM2lr1e-5_{args.prefix}'
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = True
return_representation = True

filename = f'{prefix}_{cell_line}_{seq_type}_seqlen{inp_len}_{args.folds}folds_{args.label_type}_AvgEmb{args.avg_emb}_BosEmb{args.bos_emb}_CNNlayer{args.cnn_layers}_epoch{args.epochs}_patiences{args.patience}_nodes{args.nodes}_dropout3{args.dropout3}_finetune{args.finetune}_huberloss{args.huber_loss}'
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

def train_step(train_dataloader, model, epoch):
    cnn_outputs_list, cnn_abs_ouputs_list = [], []
    relu_outputs_list = []
    linear_outputs_list, linear_abs_ouputs_list = [], []
    cnn_weight_list, cnn_abs_weight_list, cnn_grad_list, cnn_abs_grad_list = [], [], [], []
    linear_weight_list, linear_abs_weight_list, linear_grad_list, linear_abs_grad_list = [], [], [], []
        
    model.train()
    y_pred_list, y_true_list, loss_list = [], [], []
    
    for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(train_dataloader):
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
        
    loss_epoch = float(torch.Tensor(loss_list).mean())
    print(f'Train: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end = '')
    print(f'max(y_true_list) = {max(y_true_list):.4f}, max(y_pred_list) = {max(y_pred_list):.4f}')
    print(f'min(y_true_list) = {min(y_true_list):.4f}, min(y_pred_list) = {min(y_pred_list):.4f}')
    
    metrics = performances(y_true_list, y_pred_list)
    return metrics, loss_epoch, (
    loss_list, cnn_outputs_list, cnn_abs_ouputs_list, relu_outputs_list, linear_outputs_list, linear_abs_ouputs_list, cnn_weight_list, cnn_abs_weight_list, cnn_grad_list, cnn_abs_grad_list, linear_weight_list, linear_abs_weight_list, linear_grad_list, linear_abs_grad_list)


def eval_step(test_dataloader, model, epoch, data = None):
    model.eval()
    y_pred_list, y_true_list, loss_list = [], [], []
    strs_list = []
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(test_dataloader):
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
        e_pred = pd.DataFrame([strs_list, y_true_list, y_pred_list], index = [f'train_seqs', 'y_true', 'y_pred']).T
        e_pred['y_pred'] = e_pred['y_pred'].astype(np.float64)
        e_pred['y_true'] = e_pred['y_true'].astype(np.float64)
        if data is not None: 
            data_pred = pd.concat([data, e_pred], axis = 1)
        else:
            data_pred = e_pred
    return metrics, loss_epoch, data_pred

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def performances(label, pred):
    label, pred = list(label), list(pred)
    
    r = r2(label, pred)
    pearson_r = pearsonr(label, pred)[0]
    sp_cor = spearmanr(label, pred)[0]
    
    print(f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f}')
        
    return [r, pearson_r, sp_cor]

def generate_dataset_dataloader(e_data, obj_col):
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data[seq_type], mask_prob = 0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader
#######

alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

print('====Load Data====')
all_data = pd.read_csv(f'/home/ubuntu/5UTR_Optimizer-master/data/{cell_line}_sequence.csv')
test_id_list = np.load(f'/home/ubuntu/5UTR_Optimizer-master/data/{cell_line}_10foldTestID.npy', allow_pickle = True)

if args.train_atg: 
    data = all_data[all_data[seq_type].str.contains('ATG')]
if args.train_n_atg: 
    data = all_data[~all_data[seq_type].str.contains('ATG')]
else:
    data = deepcopy(all_data)

#######      
print('====Train====')
fig, axes = plt.subplots(nrows = 10, ncols = 7, figsize = (25, 40))
loss_train_repeat_dict, loss_val_repeat_dict, loss_alldata_repeat_dict = dict(), dict(), dict()
metrics_repeat_dict = dict()
best_epoch_list = []
traincv_data_pred_df = pd.DataFrame()
for fold, val_index in enumerate(test_id_list):
    print(f'====Begin Train Fold = {fold}====')
    train_index = list(set(range(len(data))).difference(set(val_index)))
    e_train = data.iloc[train_index, :].sample(frac=1, random_state = seed).reset_index(drop=True)
    e_val = data.iloc[val_index, :]
        
    train_dataset, train_dataloader = generate_dataset_dataloader(e_train, args.label_type)
    val_dataset, val_dataloader = generate_dataset_dataloader(e_val, args.label_type)

    model = CNN_linear().to(device)
    model.esm2.load_state_dict(torch.load(args.modelfile))
    if not args.finetune:
        for name, value in model.named_parameters():
            if 'esm2' in name:
                value.requires_grad = False
            print(name, value.requires_grad)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr = 1e-5, 
                                 betas = (0.9, 0.999),
                                 eps = 1e-08)

    loss_best, ep_best, r2_best = np.inf, -1, -1
    loss_train_list, loss_val_list = [], []
#     hidden_visualization_dict = dict()
    for epoch in trange(1, args.epochs + 1):

        if args.huber_loss:
            criterion = torch.nn.HuberLoss()
        else:
            criterion = torch.nn.MSELoss()
        metrics_train, loss_train, _ = train_step(train_dataloader, model, epoch)
#         hidden_visualization_dict[epoch] = hidden_visualization
        loss_train_list.append(loss_train)

        metrics_val, loss_val, _ = eval_step(val_dataloader, model, epoch)
        loss_val_list.append(loss_val)
        if args.epochs >= args.patience:
            if metrics_val[0] > r2_best: 
                path_saver = f'/home/ubuntu/esm2/Cao/saved_models/{filename}_fold{fold}.pkl'
                r2_best, ep_best = metrics_val[0], epoch
                torch.save(model.eval().state_dict(), path_saver)
                print(f'****Saving model in {path_saver}: Best epoch = {ep_best} | Train Loss = {loss_train:.4f} |  Val Loss = {loss_val:.4f} | R2_best = {r2_best:.4f}')
    
    ##### results
    best_epoch_list.append(ep_best)
        
    model_best = CNN_linear().to(device)
    model_best.load_state_dict(torch.load(path_saver))
    
    print('====ATG Analysis====')
    
    metrics_train, loss_train, train_pred = eval_step(train_dataloader, model_best, args.label_type)
    metrics_val, loss_val, val_pred = eval_step(val_dataloader, model_best, args.label_type)

    train_atg_pred = train_pred[train_pred.train_seqs.str.contains('ATG')]
    metrics_train_atg = performances(train_atg_pred['y_true'], train_atg_pred['y_pred'])
    train_n_atg_pred = train_pred[~train_pred.train_seqs.str.contains('ATG')]
    metrics_train_n_atg = performances(train_n_atg_pred['y_true'], train_n_atg_pred['y_pred']) 

    val_atg_pred = val_pred[val_pred.train_seqs.str.contains('ATG')]
    metrics_val_atg = performances(val_atg_pred['y_true'], val_atg_pred['y_pred'])
    val_n_atg_pred = val_pred[~val_pred.train_seqs.str.contains('ATG')]
    metrics_val_n_atg = performances(val_n_atg_pred['y_true'], val_n_atg_pred['y_pred']) 
    
    print('====Save y_pred====')
    val_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_valdata_fold{fold}.csv', index = False)    
    val_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_valdata_atg_fold{fold}.csv', index = False)
    val_n_atg_pred.to_csv(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_valdata_noatg_fold{fold}.csv', index = False)
    

    print('====Save Metrics====')
    metrics_repeat_dict[fold] = metrics_train + metrics_train_atg + metrics_train_n_atg + metrics_val + metrics_val_atg + metrics_val_n_atg + [loss_train.tolist()] + [loss_val.tolist()]   
    metrics_repeat_df = pd.DataFrame(metrics_repeat_dict, 
                                index = ['Train_R2', 'Train_PearsonR', 'Train_SpearmanR',
                                        'Train_R2_ATG', 'Train_PearsonR_ATG', 'Train_SpearmanR_ATG',
                                        'Train_R2_noATG', 'Train_PearsonR_noATG', 'Train_SpearmanR_noATG',
                                        'Val_R2', 'Val_PearsonR', 'Val_SpearmanR',
                                        'Val_R2_ATG', 'Val_PearsonR_ATG', 'Val_SpearmanR_ATG',
                                        'Val_R2_noATG', 'Val_PearsonR_noATG', 'Val_SpearmanR_noATG',
                                        'Loss_Train', 'Loss_Val']).T
    metrics_repeat_df['best_epoch'] = best_epoch_list
    metrics_repeat_df.fillna(-1, inplace = True)
    metrics_repeat_df.loc['mean'] = metrics_repeat_df[metrics_repeat_df['Val_SpearmanR'] != -1].mean(axis = 0)
    metrics_repeat_df.loc['std'] = metrics_repeat_df[metrics_repeat_df['Val_SpearmanR'] != -1].std(axis = 0)
    metrics_repeat_df.to_csv(f'/home/ubuntu/esm2/Cao/results/{filename}_metrics.csv', index = True)
    
    loss_train_repeat_dict[fold], loss_val_repeat_dict[fold] = [loss_train]*epoch, [loss_val]*epoch

    ##### Figures   
    axes[fold, 0].hist(train_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
    axes[fold, 0].hist(train_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
    axes[fold, 0].set_title(f'Fold{fold} Train|n={len(train_pred)}|R2={metrics_train[0]:.4f}', fontsize = 12)
    axes[fold, 0].legend(fontsize = 12)

    axes[fold, 1].hist(train_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
    axes[fold, 1].hist(train_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
    axes[fold, 1].set_title(f'Fold{fold} Train ATG|n={len(train_atg_pred)}|R2={metrics_train_atg[0]:.4f}', fontsize = 12)
    axes[fold, 1].legend(fontsize = 12)

    axes[fold, 2].hist(train_n_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
    axes[fold, 2].hist(train_n_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
    axes[fold, 2].set_title(f'Fold{fold} Train No-ATG|n={len(train_n_atg_pred)}|R2={metrics_train_n_atg[0]:.4f}', fontsize = 12)
    axes[fold, 2].legend(fontsize = 12)

    axes[fold, 3].hist(val_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
    axes[fold, 3].hist(val_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
    axes[fold, 3].set_title(f'Fold{fold} Val|n={len(val_pred)}|R2={metrics_val[0]:.4f}', fontsize = 12)
    axes[fold, 3].legend(fontsize = 12)

    axes[fold, 4].hist(val_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
    axes[fold, 4].hist(val_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
    axes[fold, 4].set_title(f'Fold{fold} Val ATG|n={len(val_atg_pred)}|R2={metrics_val_atg[0]:.4f}', fontsize = 12)
    axes[fold, 4].legend(fontsize = 12)

    axes[fold, 5].hist(val_n_atg_pred['y_true'], bins = 100, alpha = 0.3, label = f'Y_True')
    axes[fold, 5].hist(val_n_atg_pred.y_pred, bins = 100, alpha = 0.3, label = f'Y_Pred')
    axes[fold, 5].set_title(f'Fold{fold} Val No-ATG|n={len(val_n_atg_pred)}|R2={metrics_val_n_atg[0]:.4f}', fontsize = 12)

    axes[fold, 5].legend(fontsize = 12)

    
    axes[fold, 6].plot(range(args.epochs), loss_train_repeat_dict[fold], label = f'Fold_Train_{fold} = {loss_train:.4f} (Best={ep_best})', linestyle = ':')
    axes[fold, 6].plot(range(args.epochs), loss_val_repeat_dict[fold], label = f'Fold_Val_{fold} = {loss_val:.4f} (Best={ep_best})')
    axes[fold, 6].legend(fontsize = 12)
    
    plt.savefig(f'/home/ubuntu/esm2/Cao/figures/{filename}.tif')
    if args.test1fold: break
                
print(f'/home/ubuntu/esm2/Cao/saved_models/{filename}_fold{fold}.pkl')    
print(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_train_fold{fold}.csv')
print(f'/home/ubuntu/esm2/Cao/y_pred/{filename}_test_fold{fold}.csv')
print(f'/home/ubuntu/esm2/Cao/results/{filename}_metrics.csv')    
print(f'/home/ubuntu/esm2/Cao/figures/{filename}.tif')