#!/bin/bash
#SBATCH --job-name=convlstm
#SBATCH --error=model_training.log
#SBATCH --output=model_training.log
#SBATCH --nodes=1 
#SBATCH --mem=50G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1           # This basically refers to no. parallel jobs which would be run using mpi
#SBATCH --cpus-per-task=20   # How many cpus each parallel task will get
#SBATCH --gres=gpu:1         # Number of GPUs on the GPU compute node

# Load required modules (if any)
module load miniconda
module load cuda/12.0

eval "$(conda shell.bash hook)"
conda activate ug
python /scratch/IITB/monsoon_lab/24d1236/pratham/Model/main.py
sed -i 's/\r/\n/g' /scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log
