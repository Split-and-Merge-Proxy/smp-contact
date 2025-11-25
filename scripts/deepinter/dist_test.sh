set -x


CUDA_VISIBLE_DEVICES=0 python main.py --launcher pytorch --bs 1 \
                        --test --data_dir /your/own/local/path/to/deephomo \
                        --data_list_dir ./data_list/deephomo --name deepinter --output_dir ./ckpts \
                        --test_checkpoint_name 'smp_homo.pth' # smp_homo.pth for homodierms and smp_hetero.pth for heterodimers