#!/bin/bash
#
#SBATCH --job-name=Benchmark
#SBATCH --output=res.txt
#SBATCH --error=res.err
#
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gpus=1

#SBATCH --time=2:00:00

#SBATCH --mem-per-cpu=1000

srun python3 matched_comparison.py