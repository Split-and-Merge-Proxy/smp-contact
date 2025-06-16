set -x


CUDA_VISIBLE_DEVICES=0 python -u main.py --launcher pytorch --bs 1 --train --epochs 30 \
                                --data_dir /fs-computility/ai4sData/duhao.d/deephomo \
                                --data_list_dir ./data/deephomo --name deepinter \
                                --output_dir ./output_deepinter
