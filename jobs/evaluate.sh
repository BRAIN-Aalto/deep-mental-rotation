#!/bin/bash
#SBATCH --job-name=mental-rotation-model-uniform-color-evaluate
#SBATCH --time=00:20:00
#SBATCH --mem=3G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/mental-rotation-model-uniform-color-multi-gpu-evaluation.out

module load miniconda
source activate ./env
python3 evaluate_psnr.py runs/uniform-color/2023-05-14T21-47-50/best_model.pt data/ml-renderer-dataset-uniform-color/test