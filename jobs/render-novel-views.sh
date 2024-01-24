#!/bin/bash
#SBATCH -J generate-novel-views-with-pretrained-models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

module load miniconda
source activate ./env
python3 predict_w_pretrained_model.py trained-models/mountains.pt metzler_objects --save_dir gifs

