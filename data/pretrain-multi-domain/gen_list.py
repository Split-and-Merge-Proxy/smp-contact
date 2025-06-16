import os
import random

single_domain_path = '/root/projects/smp-contact/deepinter/data/pretrain-single-domain/train.txt'
ori_pre_train_list_path = '/root/projects/smp-contact/deepinter/data/pretrain/train.txt'

f = open(ori_pre_train_list_path, 'r')
ff = open(single_domain_path, 'r')

ori_pre_train_names = []
for line in f:
    ori_pre_train_names.append(line.split('\n')[0])

single_domain_names = []
for line in ff:
    single_domain_names.append(line.split('\n')[0])


multi_domain_names = list(set(ori_pre_train_names) - set(single_domain_names))
sample_multi_domain_names = multi_domain_names
# sample_multi_domain_names = random.sample(multi_domain_names, 14174)

# import pdb; pdb.set_trace()

f = open('./train.txt', 'w')

for data in sample_multi_domain_names:
    f.write(data + '\n')