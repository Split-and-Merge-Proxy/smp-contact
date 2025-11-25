To pre-process raw PDB files, follow the steps below.

## 1. Environment Setup
```bash
conda create -n pre-process python=3.8
conda activate pre-process
pip install -r requirements.txt
```

## Download the database and ESM weight
To enable MSA search and sequence feature extraction, please download the `UniRef30_2020_03` database from
https://wwwuser.gwdguser.de/~compbiol/uniclust/2020_03/,
and download the `ESM-MSA-1b` pre-trained model from
https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt.

After downloading, place the paths to the `UniRef_database` and `esm_msa_model` directories in both `gen_homo.sh` and `gen_hetero.sh`.


## 3. Pre-process Data
```bash
python -u gen_tmp_file.py
python -u gen_pkl.py
```
We also provide an example pair of PDB files in the `./example` directory to demonstrate how to convert raw PDB structures into pkl-format features.