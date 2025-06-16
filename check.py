import numpy as np
from tqdm import tqdm
import os

file_ = '/root/projects/smp-contact/deepinter/data/pretrain/train.txt'

dir_ = '/fs-computility/ai4sData/duhao.d/pseudo_multimer'

f = open(file_, 'r')

for line in tqdm(f):
    pdb_name = line.split('\n')[0]
    file_path = os.path.join(dir_, '{}.npz'.format(pdb_name))

    try:
        data = np.load(file_path, allow_pickle=True)
    except:
        print(pdb_name)