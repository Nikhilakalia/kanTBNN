#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G      
#SBATCH --time=0:15:00
#SBATCH --output=search_slurm_%j.out
#SBATCH --account=def-fslien

module load python/3.11  # Using Default Python version - Make sure to choose a version that suits your application
module load scipy-stack
source /scratch/rmcconke/training_env/bin/activate



echo "starting training..."

python3 -u hyperparam_search.py $1 > hyperparam_search_$1.log
