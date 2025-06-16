set -x

CUDA_VISIBLE_DEVICES=0 python main.py --launcher pytorch --bs 1 \
                        --train --epochs 30 --data_dir /fs-computility/ai4sData/duhao.d/data/deephomo \
                        --data_list_dir ./data/deephomo_hetero --name smp --output_dir ./output_finetune_smp_hetero --resume_checkpoint ./output_smp_new_pretrain/best.pth
