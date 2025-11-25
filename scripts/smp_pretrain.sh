set -x


CUDA_VISIBLE_DEVICES=0 python -u main.py --launcher pytorch --bs 1 --train --epochs 20 \
                                --data_dir /your/own/local/path/to/pseudo_dimer \
                                --data_list_dir ./data_list/pretrain --name smp \
                                --output_dir  /your/own/local/path/to/output