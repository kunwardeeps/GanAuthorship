#!/bin/bash
#SBATCH --job-name=image_cpu_classifier_gans
#SBATCH --output=cpu.out
#SBATCH --error=cpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kunwardeep.singh@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=24gb
#SBATCH --time=8:00:00
 
module load tensorflow/1.14
 
python Image_classifier_gans_real_cpu_binary.py
 
