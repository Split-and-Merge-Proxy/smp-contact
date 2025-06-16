import os

existing_multi_domain_dir = '/fs-computility/ai4sData/duhao.d/data/multi_domain_proteins_existing/'
ori_pre_train_list_path = '/root/projects/smp-contact/deepinter/data/pretrain_ori/train.txt'

existing_files = os.listdir(existing_multi_domain_dir)

existing_names = []
for e_file in existing_files:
    existing_names.append(e_file.split('.')[0])

ff = open(ori_pre_train_list_path, 'r')

ori_pre_train_names = []
for line in ff:
    ori_pre_train_names.append(line.split('\n')[0])

single_domain_names = []
for ori_p in ori_pre_train_names:
    if ori_p in existing_names or ori_p.split('_')[0] in existing_names:
        continue
    else:
        single_domain_names.append(ori_p)

# import pdb; pdb.set_trace()

f = open('./train.txt', 'w')

for data in single_domain_names:
    f.write(data + '\n')

