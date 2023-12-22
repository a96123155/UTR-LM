# UTR-LM: A Semi-supervised 5’ UTR Language Model for mRNA Translation and Expression Prediction

The untranslated region (UTR) of an RNA molecule plays a vital role in gene expression regulation. Specifically, the 5' UTR, located at the 5' end of an RNA molecule, is a critical determinant of the RNA’s translation efficiency. Language models have demonstrated their utility in predicting and optimizing the function of protein encoding sequences and genome sequences. In this study, we developed a semi-supervised language model for 5’ UTR, which is pre-trained on a combined library of random 5' UTRs and endogenous 5' UTRs from multiple species. We augmented the model with supervised information that can be computed given the sequence, including the secondary structure and minimum free energy, which improved the semantic representation. 

## File Structure

- Data: Store the data files or datasets required by the project. The folder can be found in [link](https://drive.google.com/drive/folders/1oGGgQ33cbx340vXsH_Ds_Py6Ad0TslLD?usp=share_link)

An example for eight MRL library:

| Training File  | Test File  | Splitting Strategy  | Descript  |
|:----------|:----------|:----------|:----------|
| 4.1_train_data_GSM3130435_egfp_unmod_1.csv    |   4.1_test_data_GSM3130435_egfp_unmod_1.csv  | Rank    |GSM3130435_egfp_unmod_1|
| 4.2_train_data_GSM3130435_egfp_unmod_1.csv    |   4.2_test_data_GSM3130435_egfp_unmod_1.csv  | Random    | GSM3130435_egfp_unmod_1|


- Scripts: Contains scripts or code files for performing specific tasks.
- Model: Contains the trained model or files related to the model.
- utrlm_requirements.txt: Lists the dependencies required by the project.

## Install

1. Create a conda environment.
```
conda create -n UTRLM python==3.9.13
conda activate UTRLM
```
2. In UTRLM environment, install the Python packages listed in the utrlm_requirements.txt file.
`pip install -r utrlm_requirements.txt`

Or
```
pip install pandas==1.4.3 
pip3 install torch torchvision torchaudio
pip install torchsummary
pip install tqdm scikit-learn scipy matplotlib seaborn
```
3. Set up Model

```bash
pip install fair-esm
find -name esm
scp -r ./Scripts/esm ./.conda/envs/UTRLM/lib/python3.9/site-packages/ # Move the folder ./Scripts/esm/ to the conda env fold, such as ./.conda/envs/UTRLM/lib/python3.9/site-packages/

```
**It is very important to Move the folder ./Scripts/esm/ to the conda env fold, such as ./.conda/envs/UTRLM/lib/python3.9/site-packages/, because we have modified the souce code of ESM.**


## Instruction Example
### UTR-LM pretraining process
```
cd ./Scripts/UTRLM_pretraining
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 v2DDP_ESM2_alldata_SupervisedInfo.py --prefix v2DDP_try --lr 1e-5 --layers 6 --heads 6 --embed_dim 16 --train_fasta ./Data/Pretrained_Data/Fivespecies_Cao_energy_structure_CaoEnergyNormalDist_255795sequence.fasta --device_ids 0 --epochs 200
```
We recommend to use the following code:

| Code File  | Decription  | 
|:----------|:----------|
| v2DDP_ESM2_alldata_SupervisedInfo.py    | MLM+MFE   | 
| v3DDP_ESM2_alldata_SupervisedInfo_SecondaryStructure.py    | MLM+MFE+SecondaryStructure   | 
| v4DDP_ESM2_alldata_SecondaryStructure.py    | MLM+SecondaryStructure   | 

Which Parameters you MUST to define:
- train_fasta: Please download the dataset from [link](https://drive.google.com/drive/u/1/folders/1_kmnYqYA5PNHQIxvwRgUn_RLZXS8Z7j3)

### UTR-LM downstream fine-tuning process
##### 1. For MRL task:
```
cd ./Scripts/UTRLM_downstream
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 5001 MJ3_Finetune_extract_append_predictor_Sample_10fold-lr-huber-DDP.py --device_ids 0,1,2,3 --label_type rl --epochs 300 --huber_loss --train_file 4.1_train_data_GSM3130435_egfp_unmod_1.csv --prefix ESM2SISS_FS4.1.ep93.1e-2.dr5 --lr 1e-2 --dropout3 0.5 --modelfile ./Model/Pretrained/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl --finetune --bos_emb --test1fold
```
Which Parameters you MUST to define:
- train_file: Please download the dataset from [link](https://drive.google.com/drive/u/1/folders/1csTXwy3LDCLKnzHHtcRsnu4LiJUEYHm3)


##### 2. For TE and EL task:
```
cd ./Scripts/UTRLM_downstream
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 9001 MJ4_Finetune_extract_append_predictor_CellLine_10fold-lr-huber-DDP.py --device_ids 0 --cell_line Muscle --label_type te_log --seq_type utr --inp_len 100 --huber_loss --modelfile ./Model/ESM2SI_3.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl --finetune --bos_emb --lr 1e-2 --dropout3 0.2 --epochs 300 --prefix TE_ESM2SI_3.1.1e-2.M.dropout2

```
Which Parameters you MUST to define:
- cell_line: Choose from "Muscle", "pc3" or "HEK". Please download the dataset from [link](https://drive.google.com/drive/u/1/folders/190oihtrwCxWjtDCK9kJzyhXPKxbr5xoR)
- label_type: Choose from "te_log" (TE task) or "rnaseq_log" (EL task)


### 3. General Parameters:
##### Important Parameters
- layers: Number of Transformer layers. We use 6. 
- heads: Number of multi-head attentions. We use 16. 
- embed_dim: Dimensions of word embeddings. We use 128.
- huber_loss: We use HuberLoss. If not choose, it will use MSELoss.
- lr: Initial learning rate. We use 1e-5 in pretraining, and 1e-2 in downstream.
- finetune: Whether to fine-tune the pre-trained LM. We fine-tuned the entire model. Noted that, if you want to fix the pre-trained model, I suggested that reduce the learning rate.
- bos_emb: We choose this as [CLS]-token embedding for downstream prediction.

##### Other Parameters
- device_ids: used to specify the ID of the GPU device used for training, the default is 2 and 3
- local-rank: DDP parameter, no need to modify
- log_interval: Interval of printing logs
- prefix: model prefix
- seq_type: sequence type (utr, utr_original_varyinglength, etc.)
- inp_len: input sequence length
- epochs: number of training rounds.
- cnn_layers: CNN layer number (default is 0), because we do not use CNN.
- avg_emb: Use the average embedding of all-token embeddings for downstream prediction, if not --avg_emb and not --bos_emb, it use all-token embeddings for downstream prediction.
- train_atg: Whether to train ATG only. Not use.
- train_n_atg: Whether to train only non-ATG. Not use.
- modelfile: model file path
- load_wholemodel: whether to load the whole model
- finetune_modeldir: the path of the finetune model
- Patience: the number of epochs used to determine the model parameter during model training
- test1fold: Whether to test only one fold of data (that is, whether to repeat the test)

### Noted
Please change the directory to your own directory.

## Reference
[Chu, Yanyi, et al. "A 5'UTR Language Model for Decoding Untranslated Regions of mRNA and Function Predictions." bioRxiv (2023): 2023-10.](https://www.biorxiv.org/content/10.1101/2023.10.11.561938v1)

## Contact
Please feel free to contact us, my email is [yanyichu@stanford.edu](yanyichu@stanford.edu).
