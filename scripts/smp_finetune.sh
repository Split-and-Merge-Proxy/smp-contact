set -x 

CUDA_VISIBLE_DEVICES=0 python main.py --launcher pytorch --bs 1 \
                        --train --epochs 30 --data_dir /your/own/local/path/to/deephomo \
                        --data_list_dir ./data_list/deephomo --name smp \
                        --output_dir /your/own/local/path/to/output \
                        --resume_checkpoint /your/own/local/path/to/pretrain ckpts