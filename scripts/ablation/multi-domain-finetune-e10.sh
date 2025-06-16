set -x


CUDA_VISIBLE_DEVICES=1 python main.py --launcher pytorch --bs 1 \
                        --train --epochs 30 --data_dir /fs-computility/ai4sData/duhao.d/data/deephomo \
                        --data_list_dir ./data/deephomo_hetero --name smp --output_dir ./output_smp_finetune_multi_domain_full_hetero_e10 --resume_checkpoint ./output_smp_pretrain_multi_domain_full/epoch_10.pth
