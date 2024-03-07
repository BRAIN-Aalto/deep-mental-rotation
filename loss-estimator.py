import argparse
import logging
import os
from pathlib import Path
import time
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from torchvision import transforms
from torchmetrics.regression import MeanSquaredError
import wandb

from models.neural_renderer import load_model
from misc.dataloaders import SameDifferentShapeDataset

import autoroot


# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


# set up argparser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt-path",
    type=lambda p: Path(p).absolute(),
    help="path to model's checkpoint"
)
parser.add_argument(
    "--data-file-path",
    type=lambda p: Path(p).absolute(),
    help="path to file with meta data about images"
)
parser.add_argument(
    "--run-name",
    default=f"run-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
    type=str,
    help="W&B run name"
)
args = parser.parse_args()

# start a wandb run
wandb.login(key=os.getenv("WANDB_KEY"))
run = wandb.init(
    project="same-different-shape-classifier",
    name=args.run_name
)

ROOT = os.getenv("PROJECT_ROOT")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# step 1: load mental-rotation model
model = load_model(args.ckpt_path).to(device)
model.eval()

# step 2: load same-different-shape dataset (validation set)
testset = SameDifferentShapeDataset(
    ROOT,
    args.data_file_path,
    transforms=transforms.Compose([
        transforms.ToTensor()
    ])
)

# step 3: define similarity metric
mse = MeanSquaredError()


def transform_and_render(
    repr,
    model,
    rotation_params,
    mirror = False
):
    """
    """
    if mirror:
        transformed = model.mirror_and_rotate(repr, rotation_params)
    else:
        transformed = model.rotate_from_source_to_target(
            repr,
            **rotation_params
        )

    rendered = model.render(transformed)

    return rendered.detach()


def log():
    """
    """
    def plot_loss(loss, ax):
        ax.set_xlim(-185, 185)
        ax.set_xticks(np.arange(-180, 180+60, 60))
        ax.plot(anglegrid, loss, color="blue")


    figure = plt.figure(figsize=(12, 5))
    grid = GridSpec(2, 4, figure=figure, hspace=0.5)

    ax11 = figure.add_subplot(grid[0, :2])

    ax12 = figure.add_subplot(grid[0, 2:], sharey=ax11)
    ax12.tick_params(axis="y", labelleft=False)

    for cond, ax in zip(["same", "different"], [ax11, ax12]):
        plot_loss(loss=list(record["loss"][cond].values()), ax=ax)

        if cond == best_cond:
            ax.axhline(best_loss, c="black", linestyle="--")
            ax.axvline(anglegrid[best], c="black", linestyle="--")


    ax2 = figure.add_subplot(grid[-1, 0])
    ax2.axis("off")
    ax2.set_title(f"source: phi = {phi_source.cpu().item():.2f}", fontsize=10)
    ax2.imshow(source.squeeze().cpu().permute(1, 2 ,0))


    ax3 = figure.add_subplot(grid[-1, 1])
    ax3.axis("off")
    ax3.set_title(f"target: phi = {phi_target:.2f}", fontsize=10)
    ax3.imshow(target.squeeze().cpu().permute(1, 2 ,0))

    
    ax4 = figure.add_subplot(grid[-1, 2])
    ax4.axis("off")
    ax4.set_title(f"matched: phi = {phi_estimated:.2f}", fontsize=10, pad=4.)

    matched = transform_and_render(
        repr=repr,
        model=model,
        rotation_params={
            "azimuth_source": phi_source,
            "elevation_source": theta_source,
            "azimuth_target": torch.tensor(phi_estimated).unsqueeze(dim=-1).to(device),
            "elevation_target": theta_source
        },
        mirror=False if best_cond == "same" else True
    )
    ax4.imshow(matched.squeeze().cpu().permute(1, 2 ,0))


    plt.close()
    wandb.log(
        {f"sample {sample_idx+1:04d}: {'same' if label else 'different'}": wandb.Image(figure)}
    )

    
# step 5: look for a rotation transformation to match two shapes
anglegrid = np.linspace(-180, 180, 181) # sampling grid

history = []

for sample_idx, sample in enumerate(testset):
    img1, img2, label = sample

    logger.info(f"Image pair {sample_idx+1:04d} ({'same' if label else 'different'}): same-different-shape task initiated.")

    target = img1.unsqueeze(0).to(device)
    source = img2.unsqueeze(0).to(device)

    # retrieve camera params used to get source image
    phi_source = torch.Tensor([testset.dataset[sample_idx]["image_2"]["azimuth"]]).to(device)
    theta_source = torch.Tensor([testset.dataset[sample_idx]["image_2"]["elevation"]]).to(device)
    # retrieve camera params used to get target image
    phi_target = testset.dataset[sample_idx]["image_1"]["azimuth"]

    # infer shape representation from source image
    repr = model.inverse_render(source)

    logger.info("Looking for a rotation angle to match two shapes ...")
    
    record = dict.fromkeys(("label", "loss"))
    
    record["loss"] = defaultdict(dict)
    record["label"] = int(label.item())

    def objective_function(rotate_by: float, mirror: bool = False):
        rendered = transform_and_render(
            repr=repr,
            model=model,
            rotation_params={
                "azimuth_source": phi_source,
                "elevation_source": theta_source,
                "azimuth_target": phi_source + torch.Tensor([rotate_by]).to(device),
                "elevation_target": theta_source
            },
            mirror=mirror
        )


        return -1 * mse(rendered, target).item()

    for angle in anglegrid:
        # first, measure similarity loss (same shape)
        record["loss"]["same"].update(
            {
                angle: objective_function(rotate_by=angle, mirror=False)
            }
        )
        # and then measure difference loss (different shape)
        record["loss"]["different"].update(
            {
                angle: objective_function(rotate_by=angle, mirror=True)
            }
        )

    losses = [max(list(record["loss"]["same"].values())), max(list(record["loss"]["different"].values()))]

    best_cond = "same" if np.argmax([losses]) == 0 else "different"

    best, best_loss = np.argmax(list(record["loss"][best_cond].values())), max(list(record["loss"][best_cond].values()))
    phi_estimated = phi_source.cpu().item() + anglegrid[best]
        
    log()

    logger.info(
        f"prediction: {best_cond}\
        mse (best) = {-best_loss:.5f}, \
        phi (correct) = {phi_target:.1f}, \
        phi (estimated) = {phi_estimated:.1f}.\n\n"
    )

    history.append(record)


with open(
    "./same-different-loss-history.json",
    "w",
    encoding="utf-8"
) as writer:
    json.dump(history, writer, indent=4)



artifact = wandb.Artifact(name="loss-history", type="data")
artifact.add_file(local_path="same-different-loss-history.json")
run.log_artifact(artifact)