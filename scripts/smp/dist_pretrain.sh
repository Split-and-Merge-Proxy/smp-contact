set -x


CUDA_VISIBLE_DEVICES=0 python -u main.py --launcher pytorch --bs 1 --train --epochs 20 \
                                --data_dir /fs-computility/ai4sData/duhao.d/pseudo_multimer \
                                --data_list_dir ./data/pretrain --name smp \
                                --output_dir ./output_smp_new_pretrain
