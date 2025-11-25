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
bash ./scripts/deepinter/dist_train.sh
```
**Note:** you can change the `data_dir`, `data_dir_list`, and `output_dir` in the Shell file to your own directory.

### SMP
```bash
bash ./scripts/smp/dist_pretrain.sh
bash ./scripts/smp/dist_finetune.sh
```
**Note:** you can change the `data_dir`, `resume_checkpoint`, and `output_dir` in the Shell file to your own directory.


## 4. Evaluations
### DeepInter
```bash
bash ./scripts/deepinter/dist_test.sh
```

### SMP
```bash
# Pytorch launcher
bash ./scripts/smp/dist_test.sh
# Slurm launcher
bash ./scripts/smp/slurm_test.sh
```

## 5. Reproducing the Results Reported in the Manuscript

To reproduce the results reported in our manuscript, first download the processed test sets (`contact_homo_test_set.zip` for homodimers and `contact_hetero_test_set.zip` for heterodimers) from https://doi.org/10.7910/DVN/0QURCP and unzip them. 

Then, change the `data_dir` in the `dist_test.sh` script to point to your local path, and run the following command:
```bash
bash ./scripts/smp/dist_test.sh
```
**Note:** you can change the `test_checkpoint_name` in the test script to use homodimer ckpt or heterodimer ckpt for test, and the DeepInter's ckpts are provided by the original paper's author.

The expected results are shown below.

**Homodimer Test Set**

|Method| P@1 | P@10 | P@25 | P@50 | P@L/10 | P@L/5 | P@L |
|----------|----------|----------|----------|----------|----------|----------|----------|
|DeepInter| 0.77 | 0.75  |  0.75 | 0.74 | 0.75 | 0.74 | 0.69 |
|SMP| 0.81  |  0.80 | 0.79  | 0.77 | 0.79 | 0.77 | 0.72 |


**Heterodimer Test Set**

|Method| P@1 | P@10 | P@25 | P@50 | P@L/10 | P@L/5 | P@L |
|----------|----------|----------|----------|----------|----------|----------|----------|
|DeepInter| 0.42 |  0.37 |  0.36 | 0.35 | 0.36 | 0.37 | 0.32 |
|SMP| 0.47  |  0.44 | 0.43  | 0.41 | 0.44 | 0.43 | 0.37 |


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