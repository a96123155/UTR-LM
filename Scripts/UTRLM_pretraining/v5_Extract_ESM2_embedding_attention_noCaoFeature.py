# CUDA_VISIBLE_DEVICES=2 python v3_Extract_ESM2_embedding_attention_CaoFeature.py --data_type Cao --data /home/ubuntu/5UTR_Optimizer-master/data/pc3_sequence_CaoFeature_withoutKmer_withoutEnergy.csv --pretrained --esm2_modelfile /home/ubuntu/esm2/saved_models/ESM2SI_3.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl --finetune --predictor_modelfile /home/ubuntu/esm2/Cao/saved_models/FeatCVESM2lr1e-5_ESM2SI_3.1_P.1e-4.dr5_pc3_te_log_27CaoFeats_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.5_finetuneTrue_huberlossTrue_magicFalse_lr0.0001_fold0_epoch234.pt --representations_bos --attentions

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
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from collections import Counter

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type = str, default = '0')
parser.add_argument('--data_type', type = str, default = 'Sample')
parser.add_argument('--data', type = str, default = '/home/ubuntu/human_5utr_modeling-master/data/4.3_test_data_GSM3130435_egfp_unmod_1.csv')
parser.add_argument('--seq_type', type = str, default = 'utr_originial_varylength')
parser.add_argument('--inp_len', type = int, default = 50)
parser.add_argument('--pretrained', action = 'store_true') ## if --pretrained: True
parser.add_argument('--esm2_modelfile', type = str, default = '/home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')

parser.add_argument('--finetune', action = 'store_true') ## if --finetune: True
parser.add_argument('--predictor_modelfile', type = str, default = '/home/ubuntu/esm2/Cao/saved_models/FeatCVESM2lr1e-5_ESM2SI_3.1_P.1e-4.dr5_pc3_te_log_27CaoFeats_utr_seqlen100_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.5_finetuneTrue_huberlossTrue_magicFalse_lr0.0001_fold0.pkl')

parser.add_argument('--logits', action = 'store_true') ## ## if --: True
parser.add_argument('--representations_pertok', action = 'store_true') ## if --: True
parser.add_argument('--representations_bos', action = 'store_true') ## if --: True
parser.add_argument('--representations_mean', action = 'store_true') ## if --: True
parser.add_argument('--attentions', action = 'store_true') ## if --: True
parser.add_argument('--attentions_symm', action = 'store_true') ## if --: True
parser.add_argument('--contacts', action = 'store_true') ## if --: True
    
args = parser.parse_args()
print(args)
global layers, heads, embed_dim, batch_toks, cnn_layers, epoch, nodes, dropout3, scaler, cao_lm, cao_bm, esm_bm, cao_esm_bm, cao_esm_lm, modelfile, magic, avg_emb, bos_emb, n_feats, inp_len
inp_len = args.inp_len
mask_prob = 0.0

modelfile = args.predictor_modelfile
output_dir = f'/home/ubuntu/esm2/{args.data_type}/Embedding_Contacts'
ESM2_results_outfilename = args.esm2_modelfile.split('/')[-1].replace('.pkl', '').replace('.pt', '')
filename = args.predictor_modelfile.split('/')[-1].replace('.pkl', '').replace('.pt', '')

cell_line = '_'.join(args.data.split('/')[-1].split('_')[:2])
pretrained_outfilename = f'Pretrained_{cell_line}__{ESM2_results_outfilename}'.replace('_6layers_16heads_128embedsize_4096batchToks_MLMLossMin', '').replace('_huberlossTrue_magicFalse', '').replace('_AvgEmbFalse', '')
finetuned_outfilename = f'Finetuned_{cell_line}__{"_".join(filename.split("/")[-1].split("_")[:10])}'.replace('_6layers_16heads_128embedsize_4096batchToks_MLMLossMin', '').replace('_huberlossTrue_magicFalse', '').replace('_AvgEmbFalse', '')#__{ESM2_results_outfilename}'
print(filename)
print(pretrained_outfilename)
print(finetuned_outfilename)

n_feats = 27

esm2_model_info = args.esm2_modelfile.split('/')[-1].split('_')
for info in esm2_model_info:
    if 'layers' in info: 
        layers = int(info[:-6])
    elif 'heads' in info:
        heads = int(info[:-5])
    elif 'embedsize' in info:
        embed_dim = int(info[:-9])
    elif 'batchToks' in info:
        batch_toks = int(info[:-9])

predictor_model_info = args.predictor_modelfile.split('_')
nodes = 40
dropout3 = 0.2
for item in predictor_model_info:
    if 'CNNlayer' in item: 
        cnn_layers = int(item[-1])
    elif 'epoch' in item:
        epoch = int(item.split('.')[0][5:])
    elif 'nodes' in item:
        nodes = int(item[5:])
    elif 'dropout3' in item:
        dropout3 = float(item[8:])

avg_emb = [False, True]['AvgEmbTrue' in modelfile]
bos_emb = [False, True]['BosEmbTrue' in modelfile]
magic = [False, True]['magicTrue' in modelfile]
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cpu'#torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

repr_layers = [0, layers]
include = ["mean"]
truncate = True
return_contacts = True
return_representation = True

class CNN_linear(nn.Module):
    def __init__(self, 
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        
        super(CNN_linear, self).__init__()
        
        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = nodes
        self.cnn_layers = cnn_layers
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        if 'SISS' in modelfile:
            print('****SISS****')
            self.esm2 = ESM2_SISS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        else:
            print('****SI****')
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
        if avg_emb or bos_emb:
            self.fc = nn.Linear(in_features = embed_dim, out_features = self.nodes)
        else:
            self.fc = nn.Linear(in_features = inp_len * embed_dim, out_features = self.nodes)
        if avg_emb or bos_emb:
            self.linear = nn.Linear(in_features = self.nbr_filters, out_features = self.nodes)
        else:
            self.linear = nn.Linear(in_features = inp_len * self.nbr_filters, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 1)
        self.direct_output = nn.Linear(in_features = embed_dim, out_features = 1)
        self.magic_output = nn.Linear(in_features = 1, out_features = 1)
        
    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation = True, return_attentions_symm = False, return_attentions = False):
        x_esm2 = self.esm2(tokens, repr_layers, need_head_weights, return_contacts, return_representation, return_attentions_symm, return_attentions)
        
        if avg_emb:
            x = x_esm2["representations"][layers][:, 1 : inp_len+1].mean(1)
            x_o = x.unsqueeze(2)
        elif bos_emb:
            x = x_esm2["representations"][layers][:, 0]
            x_o = x.unsqueeze(2)
        else:
            x_o = x_esm2["representations"][layers][:, 1 : inp_len+1]
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

        if magic:
            o = self.magic_output(o)
            
        return o, x_esm2, self.esm2  
    
alphabet = Alphabet(standard_toks = 'AGCT', mask_prob = mask_prob)
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

if args.data_type == 'Sample':
    data = pd.read_csv(args.data)
    data.rename(columns = {'rl':'label'}, inplace = True)
    dataset = FastaBatchedDataset(data.loc[:,'label'], data.utr, mask_prob = mask_prob)
elif args.data_type == 'Pretrained':
    dataset = FastaBatchedDataset.from_file(args.data, mask_prob = mask_prob)
elif args.data_type == 'Cao':
    data = pd.read_csv(args.data)
    data.rename(columns = {'te_log':'label'}, inplace = True)
    dataset = FastaBatchedDataset(data.loc[:,'label'], data[args.seq_type].str[-args.inp_len:], mask_prob = mask_prob)
else:
    data = pd.read_csv(args.data)
    dataset = FastaBatchedDataset(data.loc[:,'label'], data[args.seq_type].str[-args.inp_len:], mask_prob = mask_prob)

batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=2)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    collate_fn=alphabet.get_batch_converter(),
    batch_sampler=batches, 
    shuffle = False
)

if args.data_type == 'Pretrained': 
    max_seqlen = max([len(s[0]) for s in list(dataset)])
else:
    max_seqlen = max([len(s[1]) for s in list(dataset)])
print(f"{len(dataset)} sequences with Max SeqLen = {max_seqlen}")

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def performances(label, pred):

    r = r2(label, pred)
    pearson_r = pearsonr(label, pred)[0]
    sp_cor = spearmanr(label, pred)[0]

    print(f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f}')

    return [r, pearson_r, sp_cor]

def performances_to_pd(performances_list):
    performances_pd = pd.DataFrame(performances_list, index = ['R2', 'PearsonR', 'SpearmanR']).T
    return performances_pd

def transfer(y_prob, threshold = 0.5):
    return np.array([int(x > threshold) for x in y_prob])    


def finetuned_eval_step(test_dataloader, model, epoch, data = None):
    model.eval()
    y_pred_list, y_true_list, y_prob_list, loss_list = [], [], [], []
    
    logits_finetuned_ESM2 = []
    representations_pertok_firstLayer_finetuned_ESM2, representations_mean_firstLayer_finetuned_ESM2, representations_bos_firstLayer_finetuned_ESM2 = [], [], []
    representations_pertok_lastLayer_finetuned_ESM2, representations_mean_lastLayer_finetuned_ESM2, representations_bos_lastLayer_finetuned_ESM2 = [], [], []
    
    attentions_finetuned_ESM2, contacts_finetuned_ESM2, attentions_symm_finetuned_ESM2 = [], [], []
    labels_finetuned_ESM2, strs_finetuned_ESM2 = [], []
    i = 0
    with torch.no_grad():
        for i, (labels, strs, masked_strs, toks, masked_toks, _) in enumerate(tqdm(test_dataloader)):
            
            labels_finetuned_ESM2.extend(labels)
            strs_finetuned_ESM2.extend(strs)
            
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
                        
            outputs, results_finetuned_ESM2, _ = model(toks, need_head_weights=True, return_contacts=True, return_representation=True, return_attentions_symm = args.attentions_symm, return_attentions = args.attentions)
#             print('args.attentions = ', args.attentions)
#             print(results_finetuned_ESM2.keys())
            if args.logits: 
                try:
                    logits_finetuned_ESM2 = np.vstack([logits_finetuned_ESM2, results_finetuned_ESM2['logits'].detach().cpu().numpy()])
                except:
                    logits_finetuned_ESM2 = results_finetuned_ESM2['logits'].detach().cpu().numpy()

            if args.representations_pertok: 
                try:
                    representations_pertok_firstLayer_finetuned_ESM2 = np.vstack([representations_pertok_firstLayer_finetuned_ESM2, results_finetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].detach().cpu().numpy()])
                    representations_pertok_lastLayer_finetuned_ESM2 = np.vstack([representations_pertok_lastLayer_finetuned_ESM2, results_finetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].detach().cpu().numpy()])
                except:
                    representations_pertok_firstLayer_finetuned_ESM2 = results_finetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].detach().cpu().numpy()
                    representations_pertok_lastLayer_finetuned_ESM2 = results_finetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].detach().cpu().numpy()

            if args.representations_mean: 
                try:
                    representations_mean_firstLayer_finetuned_ESM2 = np.vstack([representations_mean_firstLayer_finetuned_ESM2, results_finetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()])
                    representations_mean_lastLayer_finetuned_ESM2 = np.vstack([representations_mean_lastLayer_finetuned_ESM2, results_finetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()])
                except:
                    representations_mean_firstLayer_finetuned_ESM2 = results_finetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()
                    representations_mean_lastLayer_finetuned_ESM2 = results_finetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()

            if args.representations_bos: 
                try:
                    representations_bos_firstLayer_finetuned_ESM2 = np.vstack([representations_bos_firstLayer_finetuned_ESM2, results_finetuned_ESM2['representations'][0][:, 0].detach().cpu().numpy()])
                    representations_bos_lastLayer_finetuned_ESM2 = np.vstack([representations_bos_lastLayer_finetuned_ESM2, results_finetuned_ESM2['representations'][layers][:, 0].detach().cpu().numpy()])
                except:
                    representations_bos_firstLayer_finetuned_ESM2 = results_finetuned_ESM2['representations'][0][:, 0].detach().cpu().numpy()
                    representations_bos_lastLayer_finetuned_ESM2 = results_finetuned_ESM2['representations'][layers][:, 0].detach().cpu().numpy()
                    

            if args.attentions:
                try:
                    attentions_finetuned_ESM2 = np.vstack([attentions_finetuned_ESM2, results_finetuned_ESM2['attentions'].detach().cpu().numpy().astype('float16')])
                except:
                    attentions_finetuned_ESM2 = results_finetuned_ESM2['attentions'].detach().cpu().numpy().astype('float16')

            if args.contacts: 
                try:
                    contacts_finetuned_ESM2 = np.vstack([contacts_finetuned_ESM2, results_finetuned_ESM2['contacts'].detach().cpu().numpy().astype('float16')])
                except:
                    contacts_finetuned_ESM2 = results_finetuned_ESM2['contacts'].detach().cpu().numpy().astype('float16')

            if args.attentions_symm:
                try:
                    new_arr = results_finetuned_ESM2['attentions_symm'].detach().cpu().numpy().astype('float16')
                    padded_ori_arr = np.pad(attentions_symm_finetuned_ESM2, ((0, 0), (0, new_arr.shape[-1] - attentions_symm_finetuned_ESM2.shape[-1]), (0, new_arr.shape[-1] - attentions_symm_finetuned_ESM2.shape[-1])), mode='constant')
                    attentions_symm_finetuned_ESM2 = np.vstack([padded_ori_arr, new_arr])
                except:
                    attentions_symm_finetuned_ESM2 = results_finetuned_ESM2['attentions_symm'].detach().cpu().numpy().astype('float16')
            y_true_list.extend(labels.cpu().reshape(-1).tolist())
            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            y_pred_list.extend(y_pred)

        print(f'Test: Epoch-{epoch} | ', end = '')
        metrics = performances(y_true_list, y_pred_list)
        if data is not None: 
            try:
                data['y_pred'] = y_pred_list
                data['y_true'] = y_true_list
            except:
                None
    print('====Save Finetuned====') 
    if args.logits: 
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/logits/{finetuned_outfilename}__logits.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 logits_ESM2 = logits_finetuned_ESM2)
        del logits_finetuned_ESM2

    if args.representations_pertok: 
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/representations_pertok/{finetuned_outfilename}__representations_pertok.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 representations_pertok_firstLayer_ESM2 = representations_pertok_firstLayer_finetuned_ESM2,
                 representations_pertok_lastLayer_ESM2 = representations_pertok_lastLayer_finetuned_ESM2)
        del representations_pertok_firstLayer_finetuned_ESM2, representations_pertok_lastLayer_finetuned_ESM2

    if args.representations_mean: 
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/representations_mean/{finetuned_outfilename}__representations_mean.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 representations_mean_firstLayer_ESM2 = representations_mean_firstLayer_finetuned_ESM2,
                 representations_mean_lastLayer_ESM2 = representations_mean_lastLayer_finetuned_ESM2)
        del representations_mean_firstLayer_finetuned_ESM2, representations_mean_lastLayer_finetuned_ESM2

    if args.representations_bos: 
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/representations_bos/{finetuned_outfilename}__representations_bos.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 representations_bos_firstLayer_ESM2 = representations_bos_firstLayer_finetuned_ESM2,
                 representations_bos_lastLayer_ESM2 = representations_bos_lastLayer_finetuned_ESM2)
        del representations_bos_firstLayer_finetuned_ESM2, representations_bos_lastLayer_finetuned_ESM2

    if args.attentions:    
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/attentions/{finetuned_outfilename}__attentions.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 attentions_ESM2 = attentions_finetuned_ESM2)
        del attentions_finetuned_ESM2

    if args.contacts:     
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/contacts/{finetuned_outfilename}__contacts.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 contacts_ESM2 = contacts_finetuned_ESM2)
        del contacts_finetuned_ESM2

    if args.attentions_symm:
        finetuned_outfilename_temp = f'{output_dir}/ESM2_finetuned/attentions_symm/{finetuned_outfilename}__attentions_symm.npz'
        print(f'Saved to: {finetuned_outfilename_temp}')
        np.savez(finetuned_outfilename_temp, 
                 sequences = strs_finetuned_ESM2,
                 labels = labels_finetuned_ESM2,
                 attentions_symm_ESM2 = attentions_symm_finetuned_ESM2)
        del attentions_symm_finetuned_ESM2

    return metrics, data#, results_finetuned_ESM2


if args.finetune: # args.data_type == IRES Sample
    # Finetuned Predictor model
    predict_model = CNN_linear().to(device)
        
    predict_model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(modelfile).items()})
    
    metrics, data = finetuned_eval_step(dataloader, predict_model, epoch, data)

    metrics = performances_to_pd(metrics)
    print(metrics)

    print('Saved to: ', f'{output_dir}/ESM2_finetuned/metrics/{finetuned_outfilename}.csv')
    print('Saved to: ', f'{output_dir}/ESM2_finetuned/e_test/{finetuned_outfilename}.csv')
    metrics.to_csv(f'{output_dir}/ESM2_finetuned/metrics/{finetuned_outfilename}.csv', index = True)
    data.to_csv(f'{output_dir}/ESM2_finetuned/e_test/{finetuned_outfilename}.csv', index = True)

# Pretrained ESM2 model
'''
data = np.load('/home/ubuntu/esm2/Embedding_Contacts/ESM2_pretrained/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.npz', allow_pickle = True)
list(data.keys())
# ['sequences', 'labels', 'representations_firstLayer_nonfinetuned_ESM2', 'representations_lastLayer_nonfinetuned_ESM2', 'attentions_nonfinetuned_ESM2', 'contacts_nonfinetuned_ESM2']
# numpy.array
'''
if args.pretrained:
    print('----Pretrained----')
    if 'SISS' in modelfile:
        ESM2_model = ESM2_SISS(num_layers = layers,
                                 embed_dim = embed_dim,
                                 attention_heads = heads,
                                 alphabet = alphabet).to(device)
    else:
        ESM2_model = ESM2(num_layers = layers,
                                 embed_dim = embed_dim,
                                 attention_heads = heads,
                                 alphabet = alphabet).to(device)
    ESM2_model.load_state_dict(torch.load(args.esm2_modelfile, map_location=device), strict = False)

#     logits_nonfinetuned_ESM2 = []
#     representations_pertok_firstLayer_nonfinetuned_ESM2, representations_mean_firstLayer_nonfinetuned_ESM2, representations_bos_firstLayer_nonfinetuned_ESM2 = [], [], []
#     representations_pertok_lastLayer_nonfinetuned_ESM2, representations_mean_lastLayer_nonfinetuned_ESM2, representations_bos_lastLayer_nonfinetuned_ESM2 = [], [], []
#     attentions_nonfinetuned_ESM2, contacts_nonfinetuned_ESM2, attentions_symm_nonfinetuned_ESM2 = [], [], []
    labels_nonfinetuned_ESM2, strs_nonfinetuned_ESM2 = [], []

#     i = 0
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(dataloader):
            labels_nonfinetuned_ESM2.extend(labels)
            strs_nonfinetuned_ESM2.extend(strs)

            toks = toks.to(device)
            results_nonfinetuned_ESM2 = ESM2_model(toks, repr_layers=repr_layers, need_head_weights=True, return_contacts=args.attentions_symm, return_representation=True, return_attentions_symm = args.attentions_symm, return_attentions = args.attentions)

            if args.logits: 
                try:
                    logits_nonfinetuned_ESM2 = np.vstack([logits_nonfinetuned_ESM2, results_nonfinetuned_ESM2['logits'].detach().cpu().numpy()])
                except:
                    logits_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['logits'].detach().cpu().numpy()

            if args.representations_pertok: 
                try:
                    representations_pertok_firstLayer_nonfinetuned_ESM2 = np.vstack([representations_pertok_firstLayer_nonfinetuned_ESM2, results_nonfinetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].detach().cpu().numpy()])
                    representations_pertok_lastLayer_nonfinetuned_ESM2 = np.vstack([representations_pertok_lastLayer_nonfinetuned_ESM2, results_nonfinetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].detach().cpu().numpy()])
                except:
                    representations_pertok_firstLayer_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].detach().cpu().numpy()
                    representations_pertok_lastLayer_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].detach().cpu().numpy()

            if args.representations_mean: 
                try:
                    representations_mean_firstLayer_nonfinetuned_ESM2 = np.vstack([representations_mean_firstLayer_nonfinetuned_ESM2, results_nonfinetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()])
                    representations_mean_lastLayer_nonfinetuned_ESM2 = np.vstack([representations_mean_lastLayer_nonfinetuned_ESM2, results_nonfinetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()])
                except:
                    representations_mean_firstLayer_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()
                    representations_mean_lastLayer_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].mean(1).detach().cpu().numpy()

            if args.representations_bos: 
                try:
                    representations_bos_firstLayer_nonfinetuned_ESM2 = np.vstack([representations_bos_firstLayer_nonfinetuned_ESM2, results_nonfinetuned_ESM2['representations'][0][:, 0].detach().cpu().numpy()])
                    representations_bos_lastLayer_nonfinetuned_ESM2 = np.vstack([representations_bos_lastLayer_nonfinetuned_ESM2, results_nonfinetuned_ESM2['representations'][layers][:, 0].detach().cpu().numpy()])
                except:
                    representations_bos_firstLayer_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['representations'][0][:, 0].detach().cpu().numpy()
                    representations_bos_lastLayer_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['representations'][layers][:, 0].detach().cpu().numpy()
                    

            if args.attentions:
                try:
                    attentions_nonfinetuned_ESM2 = np.vstack([attentions_nonfinetuned_ESM2, results_nonfinetuned_ESM2['attentions'].detach().cpu().numpy().astype('float16')])
                except:
                    attentions_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['attentions'].detach().cpu().numpy().astype('float16')

            if args.contacts: 
                try:
                    contacts_nonfinetuned_ESM2 = np.vstack([contacts_nonfinetuned_ESM2, results_nonfinetuned_ESM2['contacts'].detach().cpu().numpy().astype('float16')])
                except:
                    contacts_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['contacts'].detach().cpu().numpy().astype('float16')

            if args.attentions_symm:
                try:
                    new_arr = results_nonfinetuned_ESM2['attentions_symm'].detach().cpu().numpy().astype('float16')
                    padded_ori_arr = np.pad(attentions_symm_nonfinetuned_ESM2, ((0, 0), (0, new_arr.shape[-1] - attentions_symm_nonfinetuned_ESM2.shape[-1]), (0, new_arr.shape[-1] - attentions_symm_nonfinetuned_ESM2.shape[-1])), mode='constant')
                    attentions_symm_nonfinetuned_ESM2 = np.vstack([padded_ori_arr, new_arr])
                except:
                    attentions_symm_nonfinetuned_ESM2 = results_nonfinetuned_ESM2['attentions_symm'].detach().cpu().numpy().astype('float16')
#                     print(f"---{attentions_symm_nonfinetuned_ESM2.shape}---")
    print('====Save Pretrained====')
    if args.logits: 
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/logits/{pretrained_outfilename}__logits.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 logits_ESM2 = logits_nonfinetuned_ESM2)
        del logits_nonfinetuned_ESM2

    if args.representations_pertok: 
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/representations_pertok/{pretrained_outfilename}__representations_pertok.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 representations_pertok_firstLayer_ESM2 = representations_pertok_firstLayer_nonfinetuned_ESM2,
                 representations_pertok_lastLayer_ESM2 = representations_pertok_lastLayer_nonfinetuned_ESM2)
        del representations_pertok_firstLayer_nonfinetuned_ESM2, representations_pertok_lastLayer_nonfinetuned_ESM2

    if args.representations_mean: 
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/representations_mean/{pretrained_outfilename}__representations_mean.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 representations_mean_firstLayer_ESM2 = representations_mean_firstLayer_nonfinetuned_ESM2,
                 representations_mean_lastLayer_ESM2 = representations_mean_lastLayer_nonfinetuned_ESM2)
        del representations_mean_firstLayer_nonfinetuned_ESM2, representations_mean_lastLayer_nonfinetuned_ESM2

    if args.representations_bos: 
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/representations_bos/{pretrained_outfilename}__representations_bos.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 representations_bos_firstLayer_ESM2 = representations_bos_firstLayer_nonfinetuned_ESM2,
                 representations_bos_lastLayer_ESM2 = representations_bos_lastLayer_nonfinetuned_ESM2)
        del representations_bos_firstLayer_nonfinetuned_ESM2, representations_bos_lastLayer_nonfinetuned_ESM2

    if args.attentions:    
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/attentions/{pretrained_outfilename}__attentions.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 attentions_ESM2 = attentions_nonfinetuned_ESM2)
        del attentions_nonfinetuned_ESM2

    if args.contacts:     
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/contacts/{pretrained_outfilename}__contacts.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 contacts_ESM2 = contacts_nonfinetuned_ESM2)
        del contacts_nonfinetuned_ESM2

    if args.attentions_symm:
        nonfinetuned_outfilename_temp = f'{output_dir}/ESM2_pretrained/attentions_symm/{pretrained_outfilename}__attentions_symm.npz'
        print(f'Saved to: {nonfinetuned_outfilename_temp}')
        np.savez(nonfinetuned_outfilename_temp, 
                 sequences = strs_nonfinetuned_ESM2,
                 labels = labels_nonfinetuned_ESM2,
                 attentions_symm_ESM2 = attentions_symm_nonfinetuned_ESM2)
        del attentions_symm_nonfinetuned_ESM2


