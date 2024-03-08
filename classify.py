import argparse
import logging
import os
from pathlib import Path
import json
import time
from typing import Callable, Union, Tuple, Optional
from functools import partial
from collections import defaultdict

import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import wandb

import autoroot
from bopt import *

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
    "--iters",
    default=50,
    type=int,
    help="number of trials in the search for rotation angle to match two shapes"
)
parser.add_argument(
    "--match-threshold",
    default=0.01,
    type=float,
    help="threshold to have a match"
)
parser.add_argument(
    "--stop-threshold",
    default=None,
    type=float,
    help="threshold for the BO stopping criterion"
)
parser.add_argument(
    "--seed",
    default=12345,
    type=int,
    help="random seed"
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

rng = np.random.default_rng(seed=args.seed)

with open(
    args.data_file_path,
    "r",
    encoding="utf-8"
) as reader:
    testset = json.load(reader)


# create the camera object
camera = Camera()
# create the renderer object
renderer = Renderer(
    imgsize=(128, 128),
    dpi=100,
    bgcolor="white",
    format="png"
)

object_params = {
    "facecolor": "white",
    "edgecolor": "black",
    "edgewidth": 0.8
}


def rotate(object: Object3D, theta: float, phi: float):
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



def run_bayesian_optimization(seed: int = 12345):
    """
    """

    def log_step(plot_acq: bool = True):
        """
        """
        # figure = plt.figure(figsize=(12, 9))
        figure = plt.figure(figsize=(12, 6))
        grid = GridSpec(2, 5, figure=figure, hspace=0.5)


        def plot_GP(optim, ax):
            """
            """
            # ticks = np.linspace(0, 2*180, 5)
            # ticklabels = list(map(str, np.concatenate([np.linspace(-180, 180, 3, dtype=int), np.linspace(-180, 180, 3, dtype=int)[1:]])))
            ticks = np.linspace(0, 2*180, 17)
            ticklabels = list(map(str, np.concatenate([np.linspace(-180, 180, 9, dtype=int), np.linspace(-180, 180, 9, dtype=int)[1:]])))
            ax.set_xticks(ticks, labels=ticklabels)
            ax.xaxis.set_tick_params(labelsize=10) # 32
            ax.set_xlabel(r"$\Delta$$\theta$", fontsize=16) # 34

            # ax.set_yticks([])
            # ax.set_ylabel("Loss", fontsize=22)

            ax.set_title("Gaussian Process", fontsize=18, y=1.02) # 40

            mask = np.hstack([np.ones(int(len(points) / 2)), np.zeros(int(len(points) / 2))]).reshape(-1, 1)
            X_test = ma.array(points.reshape(-1, 1), mask=mask)

            mean, std = optim.surrogate.predict(X_test, optim.history.X, optim.history.Y, return_std=True)
            uncertainty = 1.96 * std # 95% confidence interval
            # samples = optim.surrogate.sample(
            #     X_test,
            #     X_train=optim.history.X,
            #     Y_train=optim.history.Y,
            #     n_samples=5
            # )
            
            # plot mean function
            grid = np.arange(len(points))
            ax.plot(grid, mean.ravel())
            # plot uncertainty area
            ax.fill_between(grid, (mean + uncertainty).ravel(), (mean - uncertainty).ravel(), alpha=0.1)

            # plot sampled point
            idx = int((rotate_by - (-180)) / 2)
            start = 0 if guess == "same" else 181
            ax.axvline(start + idx, c="green", linestyle="--")

            # plot training points
            if not optim.history.empty:
                for x, y, same in zip(optim.history.X.data.ravel(), optim.history.Y.ravel(), optim.history.X.mask.ravel()):
                    idx = int((x - (-180)) / 2)
                    start = 0 if same else 181
                    plt.scatter(start + idx, y, marker="X", s=35., c="black")

                # plot the current best loss value
                ax.axhline(optim.history.maximum["Y"], c="black", linestyle="--")

                # for sample in samples:
                #     ax.plot(grid, sample, lw=0.5, ls='--')


        ### Top plot: surrogate function (GP) ###
        ax1 = figure.add_subplot(grid[0, :-1])
        plot_GP(optim, ax=ax1)



        ## Bottom plot: acquisition function ###
        ax2 = figure.add_subplot(grid[1, :-1])

        if plot_acq:
            if acq is not None:
                # ticks = np.linspace(0, 2*180, 5)
                # ticklabels = list(map(str, np.concatenate([np.linspace(-180, 180, 3, dtype=int), np.linspace(-180, 180, 3, dtype=int)[1:]])))
                ticks = np.linspace(0, 2*180, 17)
                ticklabels = list(map(str, np.concatenate([np.linspace(-180, 180, 9, dtype=int), np.linspace(-180, 180, 9, dtype=int)[1:]])))
                ax2.set_xticks(ticks, labels=ticklabels)
                ax2.xaxis.set_tick_params(labelsize=10) # 32
                ax2.set_xlabel(r"$\Delta$$\theta$", fontsize=16) # 34

                # ax2.set_yticks([])
                # ax2.set_ylabel("Acquisition\nfunction", fontsize=22)

                ax2.set_title("Acquisition function", fontsize=18, y=1.02) # 40
    
                ax2.plot(
                    np.arange(len(points)),
                    acq.ravel(),
                    c="red",
                )

                idx = int((rotate_by - (-180)) / 2)
                start = 0 if guess == "same" else 181
                ax2.axvline(start + idx, c="green", linestyle="--")


        # Top-left plot: target image ###
        ax3 = figure.add_subplot(grid[0, -1])
        ax3.axis("off")
        ax3.set_title(f"phi = {phi_target:.2f}", fontsize=14)

        ax3.imshow(
            render(target, theta=theta_source, phi=phi_target)
        )


        ## Bottom-left plot: rotated image ###
        ax4 = figure.add_subplot(grid[-1, -1])
        ax4.axis("off")

        phi_estimated = phi_source + rotate_by 

        ax4.imshow(
            render(source, theta=theta_source, phi=phi_estimated)
        )
        ax4.set_title(f"phi = {phi_estimated:.2f}", fontsize=14)
        
        plt.close()
        wandb.log(
            {f"BO (sample {sample_idx:03d}: {'same' if label else 'different'})": wandb.Image(figure)}
        )


    def objective_function(rotate_by: float, mirror: bool = False):
        """
        """
        target_rotated = rotate(target, theta=theta_source, phi=phi_target)

        if mirror:
            source_mirrored = Object3D(
                shape=MetzlerShape(ShapeString(sample["image_2"]["shape"]).reflect(over=Plane(2))),
                **object_params
            )

            source_rotated = rotate(source_mirrored, theta=theta_source, phi=phi_source + rotate_by.data)
        else:
            source_rotated = rotate(source, theta=theta_source, phi=phi_source + rotate_by.data)
        
        return -LA.norm((skeleton(target_rotated) - skeleton(source_rotated)))
    
    

    # load params to set up the prior distribution
    with open(
        "prior-x-3D-skeleton-loss.json",
        "r",
        encoding="utf-8"
    ) as writer:
        params = json.load(writer)
    
    points, mean, cov = np.array(params["points"]), params["mean"], np.array(params["cov"])

    optim = BayesianOptimizer(
        obj_func=objective_function,
        search_space=LinearGridSpace(start=-180, end=180, size=181, random_state=None),
        surrogate=GaussianProcessRegressor(
            mean=MeanFunction(mean=mean),
            kernel=CustomKernel(cov=cov, points=points),
            random_state=None
        ),
        acquisition=AcquisitionFunction(eps=0.0001, kind="ei")
    )

    # stopping_criterion = SearchStoppingCriterion(kind="sampling", threshold=args.stop_threshold)
    
    def rotation_cost(X):
        cost = 1 - 0.75 * abs(X.data) / 180
        cost[~X.mask] -= 0.1 

        return cost


    for i in range(1, args.iters+1):
        guess, rotate_by, loss_value, acq = optim.step(cost_func=rotation_cost, return_acq=True)

        if DEBUG: log_step()
        optim.log({"guess": guess, "x": rotate_by, "y": loss_value}, step=i)


        if -loss_value < args.match_threshold:
            match = 1 if guess == "same" else 0
            break

    else:
        match = None

    log_step()

    return {
        "phi_estimated": phi_source + rotate_by,
        "mse_score": -loss_value,
        "prediction": match,
        "steps": i,
    }    

# step 5: look for a rotation angle to match two shapes
table = []

DEBUG = False

for sample_idx, sample in enumerate(testset):

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

    logger.info(f"Image pair {sample_idx+1:04d} ({'same' if label else 'different'}): same-different-shape task initiated.")

    # run BO to see if we can find a rotation angle (phi) to match source and target images
    logger.info("Looking for a rotation angle to match two shapes ...")


    res = run_bayesian_optimization()


    matched = render(source, theta=theta_source, phi=res["phi_estimated"])

    table.append(
        {
            "source": wandb.Image(render(source, theta=theta_source, phi=phi_source)),
            "target": wandb.Image(render(target, theta=theta_source, phi=phi_target)),
            "matched": wandb.Image(matched),
            "phi_source": phi_source,
            "phi_target": phi_target,
            **res,
            "label": label
        }
    )

    logger.info(
        f'''outcome: {'correct' if label == res["prediction"] else 'incorrect'}, \
        steps: {res["steps"]}, \
        mse = {res["mse_score"]:.5f}, \
        phi (correct) = {phi_target:.2f}, \
        phi (estimated) = {res["phi_estimated"]:.2f}.\n\n'''
    )

    if DEBUG: break



data = pd.DataFrame(table)

wandbtable = wandb.Table(data=data)
wandb.log({"result": wandbtable})

acc = 100 * np.sum(data["prediction"] == data["label"]) / len(data)
logger.info(f"Accuracy: {acc:.2f}%")
wandb.log({"acc/test": acc})