import argparse
from pathlib import Path
import time

import numpy as np
import torch
import wandb

from misc.dataloaders import scene_render_dataset
from models.neural_renderer import load_model
from misc.quantitative_evaluation import get_dataset_psnr


parser = argparse.ArgumentParser()
parser.add_argument(
    "path_to_model",
    type=lambda p: Path(p).absolute(),
    help="path to model"
)
parser.add_argument(
    "path_to_data",
    type=lambda p: Path(p).absolute(),
    help="path to folder with test images"
)
parser.add_argument(
    "--run_name",
    default=f"run-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
    type=str,
    help="W&B run name"
)


args = parser.parse_args()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

run = wandb.init(
    project="metzler-deep-model",
    name=args.run_name
)

# Get path to experiment folder from command line arguments
model_path = args.path_to_model
data_dir = args.path_to_data 

# Load model
model = load_model(model_path)
model = model.to(device)

# Initialize dataset
dataset = scene_render_dataset(
    path_to_data=data_dir,
    img_size=(3, 128, 128),
    crop_size=128,
    allow_odd_num_imgs=True
)

# Calculate PSNR
with torch.no_grad():
    psnrs = get_dataset_psnr(
        device,
        model,
        dataset,
        batch_size=50,
    )


print(f"Dataset mean PSNR: {np.mean(psnrs):.5f}")


