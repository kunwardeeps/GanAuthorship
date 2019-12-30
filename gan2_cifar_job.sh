#!/bin/bash
#SBATCH --job-name=gan2_cifar
#SBATCH --output=gan2.out
#SBATCH --error=gan2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kunwardeep.singh@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
 
module load tensorflow/1.14.0
 
python gan2_cifar.py
 
