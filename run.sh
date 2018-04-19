#!/bin/bash -l
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -t 144:00:00
#SBATCH -J transformer
#SBATCH --gres=gpu:pascal:4
#SBATCH -o training.log

hostname
echo $CUDA_VISIBLE_DEVICES
module load cuda

srun python train.py 
