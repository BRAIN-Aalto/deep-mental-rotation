import argparse
import logging
import os
from pathlib import Path
from collections import defaultdict
import time
import json

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb

from models.neural_renderer import load_model
from metzler_renderer.geometry import (
    Plane,
    ShapeString,
    MetzlerShape
)
from metzler_renderer.renderer import (
    Camera,
    Renderer,
    Object3D
)
from metzler_renderer import utils

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


# step 2: load data
with open(
    args.data_file_path,
    "r",
    encoding="utf-8"
) as reader:
    data = json.load(reader)


# create the camera object
camera = Camera()
# create the renderer object
renderer = Renderer(
    imgsize=(128, 128),
    dpi=100,
    bgcolor="white",
    format="png"
)


def rotate(
    object: Object3D,
    theta: float,
    phi: float
):
    """
    """
    camera.setSphericalPosition(
        r=25,
        theta=theta,
        phi=phi
    )
    view = camera.setLookAtMatrix()

    model = object.setModelMatrix()

    mv = view @ model

    return (mv @ utils.homogenize(object.vertices))[:-1]


def skeleton(vertices):
    """
    """
    CUBES = 10

    centroids = np.zeros((CUBES, 3))

    for i in range(CUBES):
        centroids[i, :] = np.mean(vertices[:, 8*i:8*i + 8], axis=1)

    return centroids


def render(object: Object3D, theta: float, phi: float):
    """
    """
    camera.setSphericalPosition(
        r=25,
        theta=theta,
        phi=phi
    )

    renderer.render(object, camera)

    return renderer.save_figure_to_numpy()


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
    ax2.set_title(f"source: phi = {phi_source:.2f}", fontsize=10)


    ax2.imshow(
        render(source, theta=theta_source, phi=phi_source)
    )


    ax3 = figure.add_subplot(grid[-1, 1])
    ax3.axis("off")
    ax3.set_title(f"target: phi = {phi_target:.2f}", fontsize=10)

    ax3.imshow(
        render(target, theta=theta_source, phi=phi_target)
    )


    
    ax4 = figure.add_subplot(grid[-1, 2])
    ax4.axis("off")
    ax4.set_title(f"matched: phi = {phi_estimated:.2f}", fontsize=10, pad=4.)

    ax4.imshow(
        render(source, theta=theta_source, phi=phi_estimated)
    )

    plt.close()
    wandb.log(
        {f"sample {idx+1:04d}: {'same' if label else 'different'}": wandb.Image(figure)}
    )



object_params = {
    "facecolor": "white",
    "edgecolor": "black",
    "edgewidth": 0.8
}
    
# step 5: look for a rotation transformation to match two shapes
anglegrid = np.linspace(-180, 180, 181) # sampling grid

history = []

for idx, sample in enumerate(data):

    target = Object3D(
                shape=MetzlerShape(ShapeString(sample["image_1"]["shape"])),
                **object_params
            )

    source = Object3D(
                shape=MetzlerShape(ShapeString(sample["image_2"]["shape"])),
                **object_params
            )
    
    label = sample["label"]

    phi_source = sample["image_2"]["azimuth"]

    theta_source = -sample["image_2"]["elevation"]

    phi_target = sample["image_1"]["azimuth"]


    logger.info("Looking for a rotation angle to match two shapes ...")
    
    record = dict.fromkeys(("label", "loss"))
    
    record["loss"] = defaultdict(dict)
    record["label"] = int(label)


    def objective_function(rotate_by: float, mirror: bool = False):
        """
        """
        target_rotated = rotate(target, theta=theta_source, phi=phi_target)

        if mirror:
            source_mirrored = Object3D(
                shape=MetzlerShape(ShapeString(sample["image_2"]["shape"]).reflect(over=Plane(2))),
                **object_params
            )

            source_rotated = rotate(source_mirrored, theta=theta_source, phi=phi_source + rotate_by)
        else:
            source_rotated = rotate(source, theta=theta_source, phi=phi_source + rotate_by)
        
        return -LA.norm((skeleton(target_rotated) - skeleton(source_rotated)))




    logger.info(f"Image pair {idx+1:04d} ({'same' if label else 'different'}): same-different-shape task initiated.")

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
    phi_estimated = phi_source + anglegrid[best]


    log()

    logger.info(
        f"prediction: {best_cond}\
        mse (best) = {-best_loss:.5f}, \
        phi (correct) = {phi_target:.1f}, \
        phi (estimated) = {phi_estimated:.1f}.\n\n"
    )

    history.append(record)



with open(
    "./same-different-3D-skeleton-loss-history.json",
    "w",
    encoding="utf-8"
) as writer:
    json.dump(history, writer, indent=4)



artifact = wandb.Artifact(name="loss-history", type="data")
artifact.add_file(local_path="same-different-3D-skeleton-loss-history.json")
run.log_artifact(artifact)