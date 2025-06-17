# SMP-contact 
This repository is tor protein-protein contact map prediction task

## 1. Environment Setup

```bash
git clone https://github.com/Split-and-Merge-Proxy/smp-contact.git
cd smp-contact
conda create -n smp-contact python=3.8
conda activate smp-contact
pip install -r requirements.txt
```

## 2. Data Preparation
You can download the data from [PRING - Harvard Dataverse](https://doi.org/10.7910/DVN/22AUPR) and place it in the `./data` folder.

## 3. Training (Optional)
For baseline model DeepInter
```
# Pytorch DDP
bash ./scripts/deepinter/dist_train.sh

# Slurm 
bash ./scripts/deepinter/slurm_train.sh
```
For our method SMP
```
# Pytorch
bash ./scripts/smp/dist_pretrain.sh
bash ./scripts/smp/dist_finetune.sh

# Slurm
bash ./scripts/smp/slurm_pretrain.sh
bash ./script/smp/slurm_finetune.sh
```
**Note:** Please change the ``

## 4. Evaluations
```
Pytorch
bash ./scripts/dist
```


## 5. Infernce on your custom data

