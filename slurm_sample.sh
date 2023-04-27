#!/bin/bash

#SBATCH -J edm_sample
#SBATCH -o %x_%j.out
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH -t 02:00:00
#SBATCH -A MLL

source "/home1/08134/negin/.bashrc"
source activate edm

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export CUDA_VISIBLE_DEVICES=0

NET='network-snapshot-020008'
OUTDIR="/home1/08134/negin/edm/output_imgs/$NET"
CIFAR_DIR='/home1/08134/negin/edm/cifar10_data'
NET_DIR="/home1/08134/negin/edm/output/00000-cifar10-32x32-cond-ddpmpp-edm-gpus6-batch126-fp32/$NET.pkl"

cd "/home1/08134/negin/edm/"
#mkdir $OUTDIR

srun python3 reconstruct.py --network $NET_DIR --outdir $OUTDIR --dataset "cifar10" \
     --data_dir $CIFAR_DIR 
