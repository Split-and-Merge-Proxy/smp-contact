set -x


CUDA_VISIBLE_DEVICES=1 python -u main.py --launcher pytorch --bs 1 --train --epochs 20 \
                                --data_dir /fs-computility/ai4sData/duhao.d/data/pseudo_multimer \
                                --data_list_dir ./data/pretrain-single-domain --name smp \
                                --output_dir ./output_smp_pretrain_single_domain_2