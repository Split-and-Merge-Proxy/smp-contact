set -x


for i in {1..30}
do
    echo $i 
    checkpoint=epoch_$i.pth 
    echo $checkpoint 
    CUDA_VISIBLE_DEVICES=0 python -u main.py --launcher pytorch --bs 1 \
                        --test --data_dir /fs-computility/ai4sData/duhao.d/data/deephomo \
                        --data_list_dir ./data/deephomo_hetero --name smp --output_dir ./output_smp_finetune_single_domain_hetero_best --test_checkpoint_name $checkpoint
done


