#!/bin/bash
#SBATCH --job-name=mental-rotation-model-classification-bo-baseline+sampling-stopping+linear-rotation-cost
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/same-different-shape-classification-baseline-sampling-stopping-linear-cost.out

module load miniconda
source activate ./devenv
python3 noname.py --ckpt-path runs/no-color/2023-05-14T17-44-32/best_model.pt --data-file-path ../same-different-shape-classification/data/same-different-shape-dataset/out-of-plane-rotation/1/v6/test/data_params.json --iters 50 --stop-threshold 0.5 --match-threshold 0.01