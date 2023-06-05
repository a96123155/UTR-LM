# UTR-LM: A Semi-supervised 5’ UTR Language Model for mRNA Translation and Expression Prediction

The untranslated region (UTR) of an RNA molecule plays a vital role in gene expression regulation. Specifically, the 5' UTR, located at the 5' end of an RNA molecule, is a critical determinant of the RNA’s translation efficiency. Language models have demonstrated their utility in predicting and optimizing the function of protein encoding sequences and genome sequences. In this study, we developed a semi-supervised language model for 5’ UTR, which is pre-trained on a combined library of random 5' UTRs and endogenous 5' UTRs from multiple species. We augmented the model with supervised information that can be computed given the sequence, including the secondary structure and minimum free energy, which improved the semantic representation. 

## File Structure

- Data: Store the data files or datasets required by the project. The folder can be found in https://drive.google.com/drive/folders/1hfiS40o-msRqlHlSgqh28vVlmm9cgm3C?usp=sharing
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

```
pip install fair-esm
find -name esm
scp -r ./Scripts/esm ./.conda/envs/UTRLM/lib/python3.9/site-packages/ # Move the folder ./Scripts/esm/ to the conda env fold, such as ./.conda/envs/UTRLM/lib/python3.9/site-packages/

```

## Instruction Example
### UTR-LM pretraining process
```
cd ./Scripts/UTRLM_pretraining
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 v3DDP_ESM2_alldata_SupervisedInfo_SecondaryStructure.py --prefix v3DDP_try --lr 1e-5 --layers 2 --heads 2 --embed_dim 16 --train_fasta ./Data/Pretrained_Data/Fivespecies_Cao_energy_structure_CaoEnergyNormalDist_255795sequence.fasta --device_ids 0 --epochs 2
```

- python -m torch.distributed.launch: Command to start PyTorch distributed training.
- nproc_per_node=1: Number of GPUs to use on each node.
- master_port 2345: The port number used for communication.
- v3DDP_ESM2_alldata_SupervisedInfo_SecondaryStructure.py: Python script to be run.
- prefix v3DDP_try: Prefix for saving the file name of the training result.
- lr 1e-5: Learning rate.
- layers 2: Number of Transformer layers.
- heads 2: Number of multi-head attentions.
- embed_dim 16: Dimensions of word embeddings.
- train_fasta /home/yanyi/Data/Pretrained_Data/Fivespecies_Cao_energy_structure_CaoEnergyNormalDist_255795sequence.fasta: training data.
- device_ids 0: GPU device ids to use.
- epochs 2: Number of epochs for training.

### UTR-LM downstream fine-tuning process
```
cd ./Scripts/UTRLM_downstream
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 1125 Finetune_extract_append_predictor_CellLine_10fold-lr-huber-DDP.py --device_ids 0 --prefix ESM2DSD_FC3.1.1e-2.H.dropout5 --cell_line HEK --label_type te_log --seq_type utr --inp_len 100 --lr 1e-2 --epochs 2 --huber_loss --modelfile ./Model/ESM2DSD_FC3.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_MLMLossMin_epoch201.pkl --finetune --bos_emb --dropout3 0.5

```
- device_ids: used to specify the ID of the GPU device used for training, the default is 2 and 3
- local-rank: DDP parameter, no need to modify
- log_interval: Interval of printing logs
- prefix: model prefix
- cell_line: cell line (Muscle/pc3/HEK)
- label_type: label type (te_log, rnaseq_log)
- seq_type: sequence type (utr50, utr100, etc.)
- inp_len: input sequence length
- lr: initial learning rate
- epochs: number of training rounds
- cnn_layers: CNN layer number (default is 0)
- Patience: the number of epochs used to determine the model parameter during model training
- test1fold: Whether to test only one fold of data (that is, whether to repeat the test)
- huber_loss: Whether to use Huber loss
- modelfile: model file path
- load_wholemodel: whether to load the whole model
- finetune_modeldir: the path of the finetune model
- finetune: Whether to fine-tune the pre-trained LM
- avg_emb: Use the average embedding of all-token embeddings for downstream prediction
- bos_emb: use [CLS]-token embedding for downstream prediction
- not --avg_emb and not --bos_emb: use all-token embeddings for downstream prediction
- train_atg: Whether to train ATG only
- train_n_atg: Whether to train only non-ATG

## Reference
DOI: 10.1234/utrlm

utr-lm: A Semi-supervised 5%E2%80%99 UTR Language Model for mRNA Translation and Expression Prediction
