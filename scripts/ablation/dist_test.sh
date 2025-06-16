set -x


CUDA_VISIBLE_DEVICES=0 python main.py --launcher pytorch --bs 1 \
                        --test --data_dir /fs-computility/ai4sData/duhao.d/data/deephomo \
                        --data_list_dir ./data/deephomo --name smp --output_dir ./output_smp_finetune_multi_domain_full_best --test_checkpoint_name 'best.pth'
