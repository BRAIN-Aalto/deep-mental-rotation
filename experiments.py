import argparse
import wandb
import json
import os
from pathlib import Path
import time

import autoroot
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from misc.dataloaders import (
    ShapePairMappableDataset,
    ShapePairIterableDataset,
    ShapePairBatchSampler,
    custom_collate_fn,
    ShapePairDataLoader,
    scene_render_dataloader
)
from misc.utils import seed_everything, seed_worker
from models.neural_renderer import NeuralRenderer
from training.training import Trainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "config_file",
    type=lambda p: Path(p).absolute(),
    help="path to model and training configuration file"
)
parser.add_argument(
    "--run-name",
    default=f"run-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
    type=str,
    help="W&B run name"
)
parser.add_argument(
    "--seed",
    default=12345,
    type=int,
    help="random seed"
)

args = parser.parse_args()

seed_everything(args.seed)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# start a wandb run
wandb.login(key=os.getenv("WANDB_KEY"))
run = wandb.init(
    project="metzler-deep-model",
    name=args.run_name
)

ROOT = os.getenv("PROJECT_ROOT")

# Get path to config from command line arguments
path_to_config = args.config_file

# Open config file
with open(
    path_to_config,
    "r",
    encoding="utf-8"
) as reader:
    config = json.load(reader)

# Set up directory to store experiments
timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
directory = Path(f"runs/{config['id']}/{timestamp}")
directory.mkdir(parents=True, exist_ok=True)

# Save config file in directory
with open(
    directory / "config.json",
    "w",
    encoding="utf-8"
) as writer:
    json.dump(config, writer, indent=4)


# Set up renderer
model = NeuralRenderer(
    img_shape=config["img_shape"],
    channels_2d=config["channels_2d"],
    strides_2d=config["strides_2d"],
    channels_3d=config["channels_3d"],
    strides_3d=config["strides_3d"],
    num_channels_inv_projection=config["num_channels_inv_projection"],
    num_channels_projection=config["num_channels_projection"],
    mode=config["mode"]
)

model.print_model_info()

model = model.to(device)

if config["multi_gpu"]: model = torch.nn.DataParallel(model)

g = torch.Generator()
g.manual_seed(args.seed)

# Set up train dataloader
train_dataloader = scene_render_dataloader(
    root=ROOT,
    path_to_data=config["path_to_train_data"],
    batch_size=config["batch_size"],
    img_size=config["img_shape"],
    worker_init_fn=seed_worker
)

# Optionally set up test dataloader
if config["path_to_test_data"]:
    val_dataloader = scene_render_dataloader(
        root=ROOT,
        path_to_data=config["path_to_test_data"],
        batch_size=config["batch_size"],
        img_size=config["img_shape"],
        worker_init_fn=seed_worker
)
else:
    val_dataloader = None

# Set up trainer for renderer
trainer = Trainer(
    device=device,
    model=model,
    lr=config["lr"],
    rendering_loss_type=config["loss_type"],
    ssim_loss_weight=config["ssim_loss_weight"]
)

# Train renderer, save generated images, losses and model
trainer.train(
    train_dataloader,
    config["epochs"],
    save_dir=directory,
    save_freq=config["save_freq"],
    test_dataloader=val_dataloader,
)

# Print best epoch and corresponding losses
print()
print(f"Best epoch: {np.argmin(trainer.val_loss_history['total']) + 1}")
print(f"Best train loss: {min(trainer.epoch_loss_history['total']):.5f}")
print(f"Best validation loss: {min(trainer.val_loss_history['total']):.5f}")