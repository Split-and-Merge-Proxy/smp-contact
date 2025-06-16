file_1 = './deephomo/train.txt'
file_2 = './pretrain/train.txt'

f_1 = open(file_1, 'r')
f_2 = open(file_2, 'r')

results1 = []
results2 = []
for line_1 in f_1:
    if len(line_1.split()) == 0:
        print('empty', line_1)
    results1.append(line_1.split('\n')[0])

for line_2 in f_2:
    if len(line_2.split()) == 0:
        print('empty', line_2)
    results2.append(line_2.split('\n')[0])

# import pdb; pdb.set_trace()

overlap = []
for res in results1:
    if res in results2:
        overlap.append(res)
        # print(res)

new_results2 = []
for res_2 in results2:
    if res_2 in overlap:
        new_res = res_2 + '_pre'
        new_results2.append(new_res)
    else:
        new_results2.append(res_2)


f_new = open('./train.txt', 'w')
for r in new_results2:

    f_new.write(r + '\n')

import os

for re_over in overlap:

    os.system('aws s3 mv s3://dh_data_bio/pseudo_multimer/{0}.npz s3://dh_data_bio/pseudo_multimer/{0}_pre.npz'.format(re_over))

