#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --mem=8G      
#SBATCH --time=0:5:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-fslien

module load python/3.10  # Using Default Python version - Make sure to choose a version that suits your application
source /scratch/rmcconke/training_env/bin/activate
module load scipy-stack

echo "starting training..."

python3 -u hyperparam_search.py $1 > hyperparam_search_$1.log
