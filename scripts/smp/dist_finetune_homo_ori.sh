set -x


CUDA_VISIBLE_DEVICES=1 python main.py --launcher pytorch --bs 1 \
                        --train --epochs 30 --data_dir /fs-computility/ai4sData/duhao.d/data/deephomo \
                        --data_list_dir ./data/deephomo --name smp --output_dir ./output_finetune_smp_ori_4 --resume_checkpoint ./output_pretrain_smp/epoch_10.pth
