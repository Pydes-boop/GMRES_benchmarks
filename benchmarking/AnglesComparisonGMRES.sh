#!/bin/bash
#
#SBATCH --job-name=BenchmarkGMRES
#SBATCH --output=res.txt
#SBATCH --error=res.err
#
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gpus=1

#SBATCH --time=8:00:00

#SBATCH --mem-per-cpu=2000

srun python3 angles_comparison.py
