#!/bin/bash

#SBATCH -J edm_train
#SBATCH -o %x_%j.out
#SBATCH -p gpu-a100
#SBATCH -N 2
#SBATCH -n 3
#SBATCH --cpus-per-task=43
#SBATCH -t 48:00:00
#SBATCH -A MLL

source /path/to/env

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802

DATADIR=$1
OUTDIR=$2
$INV_PROBLEM=$3

mkdir -p $OUTDIR

srun train.py --outdir=$OUTDIR \
    --data=$DATADIR --cond=1 --arch=ddpmpp --inv-problem=$INV_PROBLEM