#!/bin/bash
#SBATCH --time=02-00:00:00
#SBATCH --mem=64G
#SBATCH --job-name=mental-rotation-model-train-no-color-4xgpu
#SBATCH --gres=gpu:v100:4
#SBATCH --output=logs/reflect-and-render-model-training-multi-gpu-no-color.out

module load miniconda
source activate ./devenv
python3 experiments.py config.json

