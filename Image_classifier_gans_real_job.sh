#!/bin/bash
#SBATCH --job-name=image_classifier_gans
#SBATCH --output=ic.out
#SBATCH --error=ic.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kunwardeep.singh@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
 
module load tensorflow
 
python Image_classifier_gans_real.py
 
