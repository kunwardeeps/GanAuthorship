#!/bin/bash
#SBATCH --job-name=gan_cifar
#SBATCH --output=gan.out
#SBATCH --error=gan.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kunwardeep.singh@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
 
module load tensorflow/1.14.0
 
python gan_cifar.py
