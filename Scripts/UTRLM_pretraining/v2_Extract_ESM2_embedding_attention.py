# nohup python Extract_ESM2_embedding_attention.py --data_type Pretrained --data /home/ubuntu/esm2/data/human_transcript_5UTR_clean_30_1022.fasta --pretrained --esm2_modelfile /home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl > nohup.out 2>&1 &

# python v2_Extract_ESM2_embedding_attention.py --data_type Sample --data /home/ubuntu/human_5utr_modeling-master/data/4.2_test_data_GSM3130435_egfp_unmod_1.csv --pretrained --esm2_modelfile /home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl --finetune --predictor_modelfile ESM2_4.2.1.3.1_unmod_1_CNNlayer0_epoch25_finetuneTrue_repeat2.pkl --attentions_symm

# python Extract_ESM2_embedding_attention.py --data_type IRES --data /home/ubuntu/IRES/data/train_data_pos_neg_split0.1.csv --pretrained --esm2_modelfile /home/ubuntu/esm2/saved_models/ESM2_1.3_five_species_TrainLossMin_6layers_16heads_64embedsize_4096batchToks.pkl --finetune --predictor_modelfile ESM2_5.1.4.2.1_IRES0.1_CNNlayer3_epoch50_finetuneTrue_repeat0.pkl

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
parser.add_argument('--pretrained', action = 'store_true') ## if --pretrained: True
parser.add_argument('--esm2_modelfile', type = str, default = '/home/ubuntu/esm2/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')

parser.add_argument('--finetune', action = 'store_true') ## if --finetune: True
parser.add_argument('--predictor_modelfile', type = str, default = 'ESM2_4.3.1.3.10_unmod_1_CNNlayer0_epoch25_nodes40_dropout30.9_finetuneTrue_repeat0.pkl')

parser.add_argument('--logits', action = 'store_true') ## ## if --: True
parser.add_argument('--representations_pertok', action = 'store_true') ## if --: True
parser.add_argument('--representations_bos', action = 'store_true') ## if --: True
parser.add_argument('--representations_mean', action = 'store_true') ## if --: True
parser.add_argument('--attentions', action = 'store_true') ## if --: True
parser.add_argument('--attentions_symm', action = 'store_true') ## if --: True
parser.add_argument('--contacts', action = 'store_true') ## if --: True


args = parser.parse_args()
print(args)

mask_prob = 0.0 # 不进行蛋白质序列掩码

modelfile = args.predictor_modelfile
output_dir = f'/home/ubuntu/esm2/Embedding_Contacts'
ESM2_results_outfilename = args.esm2_modelfile.split('/')[-1].replace('.pkl', '')
filename = args.predictor_modelfile.replace('.pkl', '')

cell_line = '_'.join(args.data.split('/')[-1].split('_')[:2])
pretrained_outfilename = f'Pretrained_{cell_line}__{ESM2_results_outfilename}'
finetuned_outfilename = f'Finetuned_{cell_line}__{filename}__{ESM2_results_outfilename}'
print(filename)
print(pretrained_outfilename)
print(finetuned_outfilename)

global layers, heads, embed_dim, batch_toks, cnn_layers, epoch, nodes, dropout3

# 从模型文件名中提取一些模型参数，如layers、heads、embed_dim、batch_toks、cnn_layers、epoch、nodes、dropout3等。
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
        epoch = int(item[5:])
    elif 'nodes' in item:
        nodes = int(item[5:])
    elif 'dropout3' in item:
        dropout3 = float(item[8:])


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

repr_layers = [layers]
include = ["mean"]
truncate = True
return_contacts = True
return_representation = True

class CNN_linear_regressor(nn.Module):
    def __init__(self, 
                 border_mode='same', inp_len=50, filter_len=8, nbr_filters=120, dropout1=0, dropout2=0):
        
        super(CNN_linear_regressor, self).__init__()
        
        self.embedding_size = embed_dim # embedding维度
        self.border_mode = border_mode # 定义了填充方式，这里使用的是same填充方式，表示在原来的基础上左右各填充一个0，使得输入和输出具有相同的长度。
        self.inp_len = inp_len # 输入序列的长度
        self.nodes = nodes
        self.cnn_layers = cnn_layers
        self.filter_len = filter_len # 卷积核的长度
        self.nbr_filters = nbr_filters # 卷积核的个数
        self.dropout1 = dropout1 # dropout层的丢弃概率
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        
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
        self.fc = nn.Linear(in_features = self.inp_len * embed_dim, out_features = self.nodes)
        self.linear = nn.Linear(in_features = self.inp_len * self.nbr_filters, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 1)        
        if self.cnn_layers == -1: self.direct_output = nn.Linear(in_features = self.inp_len * embed_dim, out_features = 1)
            
    def forward(self, tokens, need_head_weights=False, return_contacts=False, return_representation = True):
        # 首先将输入的序列经过ESM2模型得到表示层的输出，然后再经过若干层的卷积操作和池化操作得到最终的输出
        x_esm2 = self.esm2(tokens, [0, layers], need_head_weights, return_contacts, return_representation)
        x = x_esm2["representations"][layers][:, 1 : self.inp_len + 1]
        x_o = x.permute(0, 2, 1)

        if self.cnn_layers >= 1:
#             print('CNN layer-1')
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2: 
#             print('CNN layer-2')
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3: 
#             print('CNN layer-3')
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)
        
        x_f = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x_f)
            else:
                o_linear = self.fc(x_f)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x_f)

        return o, x_esm2, self.esm2  
    
class CNN_linear_classifier(nn.Module):
    def __init__(self, 
                 border_mode='same', inp_len=50, filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        
        super(CNN_linear_classifier, self).__init__()
        
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
        self.fc = nn.Linear(in_features = self.inp_len * embed_dim, out_features = self.nodes)
        self.linear = nn.Linear(in_features = self.inp_len * self.nbr_filters, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 2)      
        if self.cnn_layers == -1: self.direct_output = nn.Linear(in_features = self.inp_len * embed_dim, out_features = 2)
            
    def forward(self, tokens, need_head_weights=False, return_contacts=False, return_representation = True):
        
        x_esm2 = self.esm2(tokens, [0, layers], need_head_weights, return_contacts, return_representation)
        x = x_esm2["representations"][layers][:, 1 : self.inp_len + 1]
        x_o = x.permute(0, 2, 1)

        if self.cnn_layers >= 1:
#             print('CNN layer-1')
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2: 
#             print('CNN layer-2')
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3: 
#             print('CNN layer-3')
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)
        
        x_f = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x_f)
            else:
                o_linear = self.fc(x_f)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x_f)

        return o, x_esm2, self.esm2
    
alphabet = Alphabet(standard_toks = 'AGCT', mask_prob = mask_prob)
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

if args.data_type == 'Sample':
    data = pd.read_csv(args.data, index_col = 0)
    data.rename(columns = {'rl':'label'}, inplace = True)
    dataset = FastaBatchedDataset(data.loc[:,'label'], data.utr, mask_prob = mask_prob)
elif args.data_type == 'IRES':
    data = pd.read_csv(args.data, index_col = 0).reset_index(drop = False)
    dataset = FastaBatchedDataset(data.loc[:,'label'], data.Sequence_174, mask_prob = mask_prob)
elif args.data_type == 'Pretrained':
    dataset = FastaBatchedDataset.from_file(args.data, mask_prob = mask_prob)
elif args.data_type == 'Cao':
    data = pd.read_csv(args.data)
    data.rename(columns = {'te_log':'label'}, inplace = True)
    dataset = FastaBatchedDataset(data.loc[:,'label'], data[args.seq_type], mask_prob = mask_prob)
    
batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    collate_fn=alphabet.get_batch_converter(),
    batch_sampler=batches, 
    shuffle = False
)
# print(list(dataset))
if args.data_type == 'Pretrained': 
    max_seqlen = max([len(s[0]) for s in list(dataset)])
else:
    max_seqlen = max([len(s[1]) for s in list(dataset)])
print(f"{len(dataset)} sequences with Max SeqLen = {max_seqlen}")
# print([len(b) for b in batches])


if args.data_type != 'IRES' and args.data_type != 'Pretrained': # Predictor

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
    

elif args.data_type == 'IRES':
    def performances(y_true, y_prob, threshold = 0.5, print_ = True):

        y_pred = transfer(y_prob, threshold)

        TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
        P, N = Counter(y_true)[1], Counter(y_true)[0]
        Acc_Pos = TN / N
        Acc_Neg = TP / P

        specificity = TN/(TN+FP+1.0e-4)
        sensitivity = TP/(TP+FN+1.0e-4)
        precision = TP/(TP+FP+1.0e-4)
        f1 = (2. * precision * sensitivity)/(precision + sensitivity + 1.0e-4)
        accuracy = (TP + TN) / (TP + TN + FP + FN+1.0e-4)
        mcc = ((TP*TN) - (FN*FP)) / np.sqrt(np.float64((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN)) + 1.0e-4)

        roc_auc = roc_auc_score(y_true, y_prob)
        prec, reca, _ = precision_recall_curve(y_true, y_prob)
        aupr = auc(reca, prec)

        if print_:
            print(f'Threshold = {threshold}')
            print(f'TN = {TN}, FP = {FP}, FN = {FN}, TP = {TP}')
            print(f'Acc_Pos = {Acc_Pos} | Acc_Neg = {Acc_Neg}')
            print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
            print(f'y_true: 0 = {N} | 1 = {P}')
            print(f'sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={accuracy:.4f}|mcc={mcc:.4f}')
            print(f'auc={roc_auc:.4f}|aupr={aupr:.4f}|f1={f1:.4f}|precision={precision:.4f}\n')

        return (threshold, TN, TP, FP, FN, accuracy, Acc_Pos, Acc_Neg, roc_auc, aupr, f1, mcc, sensitivity, specificity, precision)


    def performances_to_pd(performances_list):
        metrics_name = ['Threshold', 'TN', 'TP', 'FP', 'FN', 'Accuracy', 'Acc_Pos', 'Acc_Neg', 
                        'roc_auc', 'aupr', 'f1', 'mcc', 'sensitivity', 'specificity', 'precision']

        performances_pd = pd.DataFrame(performances_list, index = metrics_name).T
        
    #     performances_pd.loc['mean'] = performances_pd.mean(axis = 0)
    #     performances_pd.loc['std'] = performances_pd.std(axis = 0)

        return performances_pd



def transfer(y_prob, threshold = 0.5):
    return np.array([int(x > threshold) for x in y_prob])    

# evaluate a finetuned version of a pre-trained model on a test dataset.
def finetuned_eval_step(test_dataloader, model, epoch, data = None):
    model.eval() # sets the model to evaluation mode.
    y_pred_list, y_true_list, y_prob_list, loss_list = [], [], [], []
    
    logits_finetuned_ESM2 = []
    representations_pertok_firstLayer_finetuned_ESM2, representations_mean_firstLayer_finetuned_ESM2, representations_bos_firstLayer_finetuned_ESM2 = [], [], []
    representations_pertok_lastLayer_finetuned_ESM2, representations_mean_lastLayer_finetuned_ESM2, representations_bos_lastLayer_finetuned_ESM2 = [], [], []
    
    attentions_finetuned_ESM2, contacts_finetuned_ESM2, attentions_symm_finetuned_ESM2 = [], [], []
    labels_finetuned_ESM2, strs_finetuned_ESM2 = [], []
    # A loop over the test dataset, where it extracts the labels, sequences, and other input data for each batch. The input data is then passed through the model to generate predictions and other output values.
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(test_dataloader):
            labels_finetuned_ESM2.extend(labels)
            strs_finetuned_ESM2.extend(strs)
            
            toks = toks.to(device)
            if args.data_type == 'Sample':
                labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
            elif args.data_type == 'IRES':
                labels = torch.LongTensor(labels)
                ones = torch.sparse.torch.eye(2)
                labels = ones.index_select(0,labels).to(device)
                
            outputs, results_finetuned_ESM2,_ = model(toks, need_head_weights=True, return_contacts=True, return_representation=True)
            
            # The if statements following the model output assignment are used to selectively append certain values to the appropriate lists, depending on the configuration settings specified in the args parameter (not shown here).

            if args.logits: 
                logits_finetuned_ESM2.extend(results_finetuned_ESM2['logits'].detach().cpu().tolist())
                
            if args.representations_pertok: 
                representations_pertok_firstLayer_finetuned_ESM2.extend(results_finetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].detach().cpu().tolist())
                representations_pertok_lastLayer_finetuned_ESM2.extend(results_finetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].detach().cpu().tolist())
                
            if args.representations_mean: 
                representations_mean_firstLayer_finetuned_ESM2.extend(results_finetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].mean(1).detach().cpu().tolist())
                representations_mean_lastLayer_finetuned_ESM2.extend(results_finetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].mean(1).detach().cpu().tolist())
                
            if args.representations_bos: 
                representations_bos_firstLayer_finetuned_ESM2.extend(results_finetuned_ESM2['representations'][0][:, 0].detach().cpu().tolist())
                representations_bos_lastLayer_finetuned_ESM2.extend(results_finetuned_ESM2['representations'][layers][:, 0].detach().cpu().tolist())
                            
            if args.attentions: 
                attentions_finetuned_ESM2.extend(results_finetuned_ESM2['attentions'].detach().cpu().tolist())
                            
            if args.contacts: 
                contacts_finetuned_ESM2.extend(results_finetuned_ESM2['contacts'].detach().cpu().tolist())
                            
            if args.attentions_symm:
                attentions_symm_finetuned_ESM2.extend(results_finetuned_ESM2['attentions_symm'].detach().cpu().tolist())
                            
            if args.data_type == 'Sample':
                y_true_list.extend(labels.cpu().reshape(-1).tolist())
                y_pred = outputs.reshape(-1).cpu().detach().tolist()
                y_pred_list.extend(y_pred)
            elif args.data_type == 'IRES':
                y_true_list.extend(labels[:, 1].cpu().tolist())
                y_prob = nn.Softmax(dim = 1)(outputs)[:, 1].cpu().detach().numpy()
                y_prob_list.extend(y_prob)
#             i += 1
#             if i ==3: break
        
        if args.data_type == 'Sample': # Regression       
            scaler = preprocessing.StandardScaler()
            scaler.fit(np.array(y_true_list).reshape(-1,1))
            y_pred_unscaled_list = scaler.inverse_transform(np.array(y_pred_list).reshape(-1,1)) #Unscaled to RL
            y_pred_unscaled_list = list(y_pred_unscaled_list.reshape(-1))

            print(f'Test: Epoch-{epoch} | ', end = '')
            metrics = performances(y_true_list, y_pred_unscaled_list)
            if data is not None: 
                try:
                    data['y_pred'] = y_pred_unscaled_list
                except:
                    None
        elif args.data_type == 'IRES': # classification
            y_pred_list = transfer(y_prob_list)

            print(f'Test: Epoch-{epoch} | ', end = '')
            metrics = performances(y_true_list, y_prob_list, threshold = 0.5, print_ = True)
            if data is not None:
                try:
                    data['y_prob'] = y_prob_list
                    data['y_pred_threshold0.5'] = y_pred_list
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
    # Finetuned Predictor model 对输入的序列进行分类或回归预测
    if args.data_type == 'IRES': 
        predict_model = CNN_linear_classifier().to(device)
    else:
        predict_model = CNN_linear_regressor().to(device)
        
    predict_model.load_state_dict(torch.load(modelfile))#, map_location=lambda storage, loc: storage)) # 从modelfile中加载predict_model的参数

    metrics, data = finetuned_eval_step(dataloader, predict_model, epoch, data)

    metrics = performances_to_pd(metrics) # 将metrics转化为Pandas DataFrame格式，并打印出来。
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
if args.pretrained: # Using pre-trained model， not fine-tuned model
    ESM2_model = ESM2(num_layers = layers,
                     embed_dim = embed_dim,
                     attention_heads = heads,
                     alphabet = alphabet).to(device)
    ESM2_model.load_state_dict(torch.load(args.esm2_modelfile), strict = False)

    logits_nonfinetuned_ESM2 = []
    representations_pertok_firstLayer_nonfinetuned_ESM2, representations_mean_firstLayer_nonfinetuned_ESM2, representations_bos_firstLayer_nonfinetuned_ESM2 = [], [], []
    representations_pertok_lastLayer_nonfinetuned_ESM2, representations_mean_lastLayer_nonfinetuned_ESM2, representations_bos_lastLayer_nonfinetuned_ESM2 = [], [], []
    attentions_nonfinetuned_ESM2, contacts_nonfinetuned_ESM2, attentions_symm_nonfinetuned_ESM2 = [], [], []
    labels_nonfinetuned_ESM2, strs_nonfinetuned_ESM2 = [], []

#     i = 0
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(dataloader):
            labels_nonfinetuned_ESM2.extend(labels)
            strs_nonfinetuned_ESM2.extend(strs)

            toks = toks.to(device)
            results_nonfinetuned_ESM2 = ESM2_model(toks, repr_layers=[0, layers], need_head_weights=True, return_contacts=True, return_representation=True)

            if args.logits: 
                logits_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['logits'].detach().cpu().tolist())

            if args.representations_pertok: 
                representations_pertok_firstLayer_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].detach().cpu().tolist())
                representations_pertok_lastLayer_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].detach().cpu().tolist())

            if args.representations_mean: 
                representations_mean_firstLayer_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['representations'][0][:, 1 : max_seqlen + 1].mean(1).detach().cpu().tolist())
                representations_mean_lastLayer_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['representations'][layers][:, 1 : max_seqlen + 1].mean(1).detach().cpu().tolist())

            if args.representations_bos: 
                representations_bos_firstLayer_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['representations'][0][:, 0].detach().cpu().tolist())
                representations_bos_lastLayer_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['representations'][layers][:, 0].detach().cpu().tolist())

            if args.attentions: 
                attentions_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['attentions'].detach().cpu().tolist())

            if args.contacts: 
                contacts_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['contacts'].detach().cpu().tolist())

            if args.attentions_symm:
                attentions_symm_nonfinetuned_ESM2.extend(results_nonfinetuned_ESM2['attentions_symm'].detach().cpu().tolist())
#             i += 1
#             if i ==3: break
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

