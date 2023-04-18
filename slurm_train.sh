#!/bin/bash

#SBATCH -J edm_train
#SBATCH -o %x_%j.out
#SBATCH -p gpu-a100
#SBATCH -N 2
#SBATCH --tasks-per-node=3
#SBATCH --cpus-per-task=43
#SBATCH -t 48:00:00
#SBATCH -A MLL

source /work2/08002/gsmyrnis/frontera/conda/miniconda3/bin/activate edm

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=8080

DATADIR=$1
OUTDIR=$2
INV_PROBLEM=$3

cd /work2/08002/gsmyrnis/ls6/neurips2023/diffusion/edm

mkdir -p $OUTDIR

srun python train.py --outdir=$OUTDIR \
    --data=$DATADIR --cond=1 --arch=ddpmpp --inv-problem=$INV_PROBLEM
