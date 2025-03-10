#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=same-different-shape-classifier-model=resnet18,dataset=depth-rotation,transforms=normalize,loss=bce
#SBATCH --gres=gpu:v100:1

module load mamba
source activate ./devenv
python3 src/train.py 'experiment="model=resnet18,dataset=depth-rotation,curriculum-learning,max-angle=180"' 'ckpt_path="runs/model=resnet18,dataset=depth-rotation,curriculum-learning,loss=bce,seed=12345/2025-01-22T10-43-17/train/ckpts/ckpt.pth"'