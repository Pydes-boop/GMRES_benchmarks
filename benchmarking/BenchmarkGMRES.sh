#!/bin/bash
#
#SBATCH --job-name=BenchmarkGMRES
#SBATCH --output=res.txt
#SBATCH --error=res.err
#
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gpus=1

#SBATCH --time=12:00:00

#SBATCH --mem-per-cpu=1000

srun python3 filter_comparison_analytical.py
srun python3 angles_comparison.py
srun python3 matched_comparison_2D.py
srun python3 matched_comparison_3D.py
