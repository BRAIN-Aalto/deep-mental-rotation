#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=same-different-shape-classifier-model=resnet18,dataset=depth-rotation,transforms=normalize,loss=bce
#SBATCH --gres=gpu:v100:1

module load mamba
source activate ./devenv
python3 src/eval.py 'experiment="model=resnet18,dataset=depth-rotation,curriculum-learning,max-angle=180"' 'ckpt_path="runs/model=resnet18,dataset=depth-rotation,curriculum-learning,loss=bce,seed=12345/2025-01-22T11-16-49/train/ckpts/ckpt.pth"'