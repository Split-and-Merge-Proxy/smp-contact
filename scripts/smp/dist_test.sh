set -x


CUDA_VISIBLE_DEVICES=0 python main.py --launcher pytorch --bs 1 \
                        --test --data_dir /fs-computility/ai4sData/duhao.d/data/deephomo \
                        --data_list_dir ./data/deephomo --name smp --output_dir /fs-computility/ai4sData/duhao.d/ckpts/smp_ckpts/output_finetune_smp_best_torch1.7_3 --test_checkpoint_name 'best.pth'
