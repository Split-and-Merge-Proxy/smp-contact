set -x

export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

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
    python main.py --launcher slurm --bs 1 --train --epochs 30 --s3_dir s3://dh_data_bio/deephomo --data_dir ./data/deephomo_hetero --name smp --output_dir ./output_finetune_smp_pre10_torch1.7.1_hetero_2 --resume_checkpoint ./output_pretrain_smp_new/epoch_10.pth
