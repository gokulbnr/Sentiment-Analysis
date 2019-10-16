#!/bin/bash
#######################################
# Script for sbatch (Slurm batch run) #
########################################

#SBATCH -A soumya.v 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anurudh.peduri@research.iiit.ac.in

module add cuda/8.0
module add cudnn/7-cuda-8.0

# wordvec generation
python make.py preprocess --dataset=aclImdb --parser=aclImdb --model=aclImdb --gpu

# training
python make.py train --dataset=aclImdb --parser=aclImdb --model=aclImdb --output=run_25_11_01 --gpu

# testing
python make.py test --dataset=aclImdb --parser=aclImdb --model=aclImdb --load-from=run_25_11_01_final --gpu
