#!/bin/bash

#SBATCH -J edm_peft_train
#SBATCH -o %x_%j.out
#SBATCH -p gpu-a100
#SBATCH -N 2
#SBATCH -n 3
#SBATCH --cpus-per-task=43
#SBATCH -t 48:00:00
#SBATCH -A MLL

source "/home1/08134/negin/.bashrc"
source activate edm

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802

DATADIR='/home1/08134/negin/edm/datasets/cifar10-32x32.zip'
OUTDIR='/home1/08134/negin/edm/output_peft'
INV_PROBLEM='inpainting'

cd "/home1/08134/negin/edm/"
mkdir -p $OUTDIR

srun python3 train.py --outdir=$OUTDIR  --data=$DATADIR \
    --cond=1 --arch=ddpmpp --inv-problem=$INV_PROBLEM --batch 128 \
    --peft --pretrained="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"