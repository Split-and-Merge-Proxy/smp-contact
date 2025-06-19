# SMP-contact 
This repository is for the protein-protein contact map prediction task.

## 1. Environment Setup

```bash
git clone https://github.com/Split-and-Merge-Proxy/smp-contact.git
cd smp-contact
conda create -n smp-contact python=3.8
conda activate smp-contact
pip install -r requirements.txt
```

## 2. Data Preparation
You can download the contact data from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/22AUPR) and place it in the `./data` folder.

## 3. Training (Optional)
### DeepInter
```bash
# Pytorch DDP
bash ./scripts/deepinter/dist_train.sh

# Slurm 
bash ./scripts/deepinter/slurm_train.sh
```
**Note:** you can change the `data_dir`, `data_dir_list`, and `output_dir` in the Shell file to your own directory.

### SMP
```bash
# Pytorch launcher
bash ./scripts/smp/dist_pretrain.sh
bash ./scripts/smp/dist_finetune.sh

# Slurm launcher
bash ./scripts/smp/slurm_pretrain.sh
bash ./script/smp/slurm_finetune.sh
```
**Note:** you can change the `data_dir`, `data_dir_list`, `resume_checkpoint`, and `output_dir` in the Shell file to your own directory.



## 4. Evaluations
### DeepInter
```bash
# Pytorch launcher
bash ./scripts/deepinter/dist_test.sh
# Slurm launcher
bash ./scripts/deepinter/slurm_test.sh
```

### SMP
```bash
# Pytorch launcher
bash ./scripts/smp/dist_test.sh
# Slurm launcher
bash ./scripts/smp/slurm_test.sh
```


## 5. Infernce on your custom data
We have already uploaded the trained weights of SMP in the `./ckpts`, you can directly download it and place it in your own directory.
```bash
python -u custom_inference.py
```

## Acknowledges
- [DeepInter](http://huanglab.phys.hust.edu.cn/DeepInter/)
- [DeepInteract](https://github.com/BioinfoMachineLearning/DeepInteract)
- [AlphaFold2](https://github.com/google-deepmind/alphafold)
- [graphtransformer](https://github.com/graphdeeplearning/graphtransformer)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)