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
You can download the contact data from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/JGDBTB) and place it in the `./data` folder.

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

## 5. Reproduce the results reported in the manuscript
To reproduce the results reportted in the our manuscript, you should first download the processed test set from the XXXXX and palced them to the `./test_set` directorny and change the `data_dir` in the `dist_test.sh` script.
The excepted results are as follows:

for the homodimer test set

| P@1 | P@10 | P@25 | P@50 | P@L/10 | P@L/5 | P@L |
|----------|----------|----------|----------|----------|----------|----------|
| 0.81  |  0.80 | 0.79  | 0.77 | 0.79 | 0.77 | 0.72 |


for the hetrodimer test set

| P@1 | P@10 | P@25 | P@50 | P@L/10 | P@L/5 | P@L |
|----------|----------|----------|----------|----------|----------|----------|
| 0.47  |  0.44 | 0.43  | 0.41 | 0.44 | 0.43 | 0.37 |


## 6. Infernce on your custom data
We have already uploaded the trained weights of SMP in the `./ckpts`, you can directly download it and place it in your own directory.
Additionally, we offer a preprocessing script ([preprocess](https://github.com/Split-and-Merge-Proxy/smp-contact/tree/main/preprocess)) that directly converts raw PDB files into pkl‐format input features.
```bash
python -u custom_inference.py
```
The output should be a NumPy-format contact map, saved as `contact_map.npy`.

## Acknowledges
- [DeepInter](http://huanglab.phys.hust.edu.cn/DeepInter/)
- [DeepInteract](https://github.com/BioinfoMachineLearning/DeepInteract)
- [AlphaFold2](https://github.com/google-deepmind/alphafold)
- [graphtransformer](https://github.com/graphdeeplearning/graphtransformer)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)