set -x

CUDA_VISIBLE_DEVICES=0 python -u main.py --launcher pytorch --bs 1 --train --epochs 30 \
                                --data_dir /your/own/local/path/to/deephomo \
                                --data_list_dir ./data_list/deephomo --name deepinter \
                                --output_dir /your/own/local/path/to/output