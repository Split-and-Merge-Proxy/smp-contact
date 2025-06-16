import torch
import os
import io
import numpy as np
import math
import random
from torch.utils.data import Dataset
from utils import read_txt


class DeepHomoDataset(Dataset):
    
    def __init__(self, s3_dir, data_dir, data_list_dir, launcher, data_ratio, max_seq_len, mode):
        
        file_ = os.path.join(data_list_dir, '{}.txt'.format(mode))
        self.file_frames = read_txt(file_)
        if data_ratio != 1.0:
            self.file_frames = random.sample(self.file_frames, math.floor(data_ratio * len(self.file_frames)))
        self.max_seq_len = max_seq_len
        self.s3_dir = s3_dir
        self.data_dir = data_dir
        self.mode = mode
        self.launcher = launcher

    def __getitem__(self, idx):

        self.pdb_name = self.file_frames[idx]

        if self.launcher == 'slurm':
            from petrel_client.client import Client
            client = Client('~/petreloss_v2.conf')
            data_file = os.path.join(self.s3_dir, self.file_frames[idx]+'.npz')
            data = np.load(io.BytesIO(client.get(data_file, enable_cache=True)), allow_pickle=True)

        elif self.launcher == 'pytorch':
            data_file = os.path.join(self.data_dir, self.file_frames[idx]+'.npz')
            data = np.load(data_file, allow_pickle=True)
        else:
            raise ValueError("Unknown job launcher {}!!!".format(self.launcher))    

        
        
        
        data = dict(data)
        data['pdb_name'] = self.pdb_name
        if self.mode == 'train':
            processed_data = self.process_data(data)
            return processed_data
        else:
            return data

    def __len__(self):
        return len(self.file_frames)
    
    def scan_sequence(self, data, kernel, l_seq_len, r_seq_len):

        stride = 1
        kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
        contact_map = data['contact_map']

        output_h, output_w = math.floor(l_seq_len - kernel_h), math.floor(r_seq_len - kernel_w)
        ij_max = 0
        for i in range(0, output_h+1, stride):
            for j in range(0, output_w+1, stride):
                sum_ = np.sum(contact_map[i: i+kernel_h, j: j+kernel_w] * kernel)
                if sum_ > ij_max:
                    i_max, j_max = i, j
                    ij_max = sum_
                else:
                    i_max, j_max = random.randint(0, output_h+1), random.randint(0, output_w+1)

        data['rec1d'] = data['rec1d'][:, i_max: i_max+kernel_h]    # [input_channels, seq_len]
        data['rec2d'] = data['rec2d'][:, i_max: i_max+kernel_h, i_max: i_max+kernel_h]   # [input_channels, seq_len, seq_len]
        data['lig1d'] = data['lig1d'][:, j_max: j_max+kernel_w]
        data['lig2d'] = data['lig2d'][:, j_max: j_max+kernel_w, j_max: j_max+kernel_w]
        data['com2d'] = data['com2d'][:, i_max: i_max+kernel_h, j_max: j_max+kernel_w]   
        data['intra_distA'] = data['intra_distA'][:, i_max: i_max+kernel_h, i_max: i_max+kernel_h]
        data['intra_distB'] = data['intra_distB'][:, j_max: j_max+kernel_w, j_max: j_max+kernel_w]
        data['contact_map'] = data['contact_map'][i_max: i_max+kernel_h, j_max: j_max+kernel_w]
        data['flatten_contact_map'] = data['contact_map'].flatten()
        return data
    
    def process_data(self, data):
        
        seqA, seqB = data['seqA'].tolist(), data['seqB'].tolist()
        l_seq_len, r_seq_len = len(seqA), len(seqB)
        if l_seq_len > self.max_seq_len or r_seq_len > self.max_seq_len:
            kernel = np.ones([min(l_seq_len, self.max_seq_len), min(r_seq_len, self.max_seq_len)])
            processed_data = self.scan_sequence(data, kernel, l_seq_len, r_seq_len) 
            # processed_data = self.tensor_Variable(processed_data)     
        else:
            processed_data = data
            # processed_data = self.tensor_Variable(processed_data)
        
        return processed_data
        
        
