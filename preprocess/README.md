To preprocess raw PDB files into pkl-format input features, please follow the steps below.

## 1. Environment Setup
```bash
conda create -n pre-process python=3.8
conda activate pre-process
pip install -r requirements.txt
```

## 2. Download the Database and ESM Weights
To enable MSA search and sequence feature extraction, please download the `UniRef30_2020_03` database from
https://wwwuser.gwdguser.de/~compbiol/uniclust/2020_03/,
and download the `ESM-MSA-1b` pre-trained model from
https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt.

After downloading, change the `UniRef_database` and `esm_msa_model` in both `gen_homo.sh` and `gen_hetero.sh` scripts to your own paths.


## 3. Data Pre-processing
```bash
python -u gen_tmp_file.py   # generate a series of intermediate feature files
python -u gen_pkl.py        # convert the feature files into a pkl-format file
```
We also provide an example pair of PDB files in the `./example` directory to demonstrate how to convert raw PDB structures into pkl-format features.