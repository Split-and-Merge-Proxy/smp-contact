set -x

PARTITION=$1
NODES=$2
JOB_NAME=$3
GPUS=$4
GPUS_PER_NODE=$5
CPUS_PER_TASK=$6
QUOTATYPE=${QUOTATYPE:-'reserved'}

srun -p ${PARTITION} \
    --nodes=${NODES} \
    --job-name=${JOB_NAME} \
    --quotatype=${QUOTATYPE} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    bash -c '
    for i in {1..30}
    do
        echo $i 
        checkpoint=epoch_$i.pth 
        echo $checkpoint 
        python main.py --launcher slurm --bs 1 --test --s3_dir s3://dh_data_bio/deephomo --data_dir ./data/deephomo_hetero --test_checkpoint_name $checkpoint --output_dir './output_finetune_smp_pre10_torch1.7.1_hetero' --name smp
    done
    '
