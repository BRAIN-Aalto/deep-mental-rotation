import argparse
import logging
import os
from pathlib import Path
import time
from typing import Callable, Union, Tuple, Optional
from functools import partial

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from torch import nn
from torchvision import transforms, utils
import wandb

from models.neural_renderer import load_model
from misc.dataloaders import SameDifferentShapeDataset

import autoroot
from bopt import *


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

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# step 1: load mental-rotation model
model = load_model(args.ckpt_path).to(device)
model.eval()

# step 2: load same-different-shape dataset (test set)
testset = SameDifferentShapeDataset(
    ROOT,
    args.data_file_path,
    transforms=transforms.Compose([
        transforms.ToTensor()
    ])
)

# step 3: define similarity metric
mse = nn.MSELoss()


def transform_and_render(
    repr,
    model,
    rotation_params,
    mirror = False
):
    """
    """
    if mirror:
        rotated = model.mirror_and_rotate(repr, rotation_params)
    else:
        rotated = model.rotate_from_source_to_target(
            repr,
            **rotation_params
        )

    rendered = model.render(rotated)

    return rendered.detach()


def run_bayesian_optimization():
    """
    """
    def log_step(plot_acq: bool = True):
        """
        """
        figure = plt.figure(figsize=(6, 5))
        grid = GridSpec(2, 3, figure=figure, hspace=0.3)

        ### Top plot: surrogate function (GP) ###
        ax1 = figure.add_subplot(grid[0, :-1])
        
        ax1.set_xlim(-180 - 5, 180 + 5)
        ax1.set_xticks(np.linspace(-180, 180, 9))

        X_test = np.linspace(-180, 180, 200).reshape(-1, 1)

        mean, std = boptim.surrogate.predict(X_test, boptim.history.X, boptim.history.Y, return_std=True)
        uncertainty = 1.96 * std # 95% confidence interval
        samples = boptim.surrogate.sample(
            X_test,
            X_train=boptim.history.X,
            Y_train=boptim.history.Y,
            n_samples=5
        )
        
        # plot mean function
        ax1.plot(X_test.ravel(), mean.ravel())
        # plot uncertainty area
        plt.fill_between(X_test.ravel(), (mean + uncertainty).ravel(), (mean - uncertainty).ravel(), alpha=0.1)

        # plot the sampled point
        ax1.axvline(phi_delta, c="green", linestyle="--")

        # plot training points
        if boptim.history.X is not None:
            ax1.plot(
                boptim.history.X.ravel(),
                boptim.history.Y.ravel(), 
                "kX",
                markersize=5,
                markeredgewidth=0.3
            )

            # plot the current best loss value
            ax1.axhline(boptim.history.maximum["Y"], c="black", linestyle="--")

            for sample in samples:
                ax1.plot(X_test, sample, lw=0.5, ls='--')


        ### Bottom plot: acquisition function ###
        ax2 = figure.add_subplot(grid[1, :-1])

        ax2.set_xlim(-180 - 5, 180 + 5)
        ax2.set_xticks(np.linspace(-180, 180, 9))

        if plot_acq:

            X_tries = boptim.space.sample(mode="fixed")

            if ei is not None:
                # plot 
                ax2.plot(
                    X_tries,
                    ei.ravel(),
                    c="red",
                )

                # plot acquisition function values at sampled points
                ax2.axvline(phi_delta, c="green", linestyle="--")


        ### Top-left plot: target image ###
        ax3 = figure.add_subplot(grid[0, -1])

        ax3.imshow(target.squeeze().cpu().permute(1, 2 ,0))
        ax3.axis("off")
        ax3.set_title(f"phi = {phi_target:.2f}", fontsize=10)


        ### Bottom-left plot: rotated image ###
        ax4 = figure.add_subplot(grid[-1, -1])

        phi_estimated = phi_source + \
              torch.tensor(boptim.history.X[-1]).to(device) if boptim.history.X is not None else phi_source
        rendered = transform_and_render(
            repr=repr,
            model=model,
            rotation_params={
                "azimuth_source": phi_source,
                "elevation_source": theta_source,
                "azimuth_target": phi_estimated,
                "elevation_target": theta_source
            },
            mirror=False
        )
        ax4.imshow(rendered.squeeze().cpu().permute(1, 2 ,0))
        ax4.axis("off")
        ax4.set_title(f"phi = {phi_estimated.squeeze().cpu():.2f}", fontsize=10)


        plt.close()
        wandb.log(
            {f"BO (sample {sample_idx:03d}: {'same' if label else 'different'})": wandb.Image(figure)}
        )


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

        # compute l2-distance between target (img1) and rendered (rotated img2)
        return -1 * mse(rendered, target)


    # same
    boptim = BayesianOptimizer(
        obj_func=objective_function,
        search_space=LinearGridSpace(random_state=None), # LogGridSpace(start=-180, end=180, size=50),
        surrogate=GaussianProcessRegressor(
            mean=MeanFunction(mean_value=-0.0332),
            kernel=CustomKernel(file="kernel.npy", points=np.linspace(-180, 180, 200)),
            random_state=None
        ),
        acquisition=AcquisitionFunction(eps=0.0001, kind="ei")
    )

    stopping_criterion = SearchStoppingCriterion(kind="sampling", threshold=args.stop_threshold)

    # rotation_cost = lambda rotate_by: 1 - 0.75 * abs(rotate_by) / 180 # optional

    for step in range(1, args.iters+1):
        phi_delta, loss_value, ei = boptim.step(cost_func=None, return_acq=True)

        log_step()

        boptim.log({"x": phi_delta, "y": loss_value}, step=step)

        if stopping_criterion(boptim, X=np.linspace(-180, 180, 200).reshape(-1, 1)): break

    
    if -boptim.history.maximum["Y"] > args.match_threshold:
        match = False
    else:
        match = True


    log_step(plot_acq=False)

    return {
        "phi_estimated": phi_source.item() + boptim.history.maximum["X"],
        "mse_score": -boptim.history.maximum["Y"],
        "steps": step,
        "prediction": int(match)
    }    



# step 5: look for a rotation angle to match two shapes
table = []

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

    # run BO to see if we can find a rotation angle (phi) to match source and target images
    logger.info("Looking for a rotation angle to match two shapes ...")

    res = run_bayesian_optimization()

    matched = transform_and_render(
        repr=repr,
        model=model,
        rotation_params={
            "azimuth_source": phi_source,
            "elevation_source": theta_source,
            "azimuth_target": torch.tensor(res["phi_estimated"]).unsqueeze(dim=-1).to(device),
            "elevation_target": theta_source
        },
        mirror=False
    )

    table.append(
        {
            "source": wandb.Image(source.squeeze().cpu().permute(1, 2, 0).numpy()),
            "target": wandb.Image(target.squeeze().cpu().permute(1, 2, 0).numpy()),
            "matched": wandb.Image(matched.squeeze().cpu().permute(1, 2, 0).numpy()),
            "phi_source": phi_source.squeeze().cpu(),
            "phi_target": phi_target,
            **res,
            "label": label.item()
        }
    )

    logger.info(
        f'''outcome: {'correct' if label == res["prediction"] else 'incorrect'}, \
        steps: {res["steps"]}, \
        mse (best) = {res["mse_score"]:.5f}, \
        phi (correct) = {phi_target:.2f}, \
        phi (estimated) = {res["phi_estimated"]:.2f}.'''
    )


data = pd.DataFrame(table)

wandbtable = wandb.Table(data=data)
wandb.log({"result": wandbtable})

acc = 100 * np.sum(data["prediction"] == data["label"]) / len(data)
logger.info(f"Accuracy: {acc:.2f}")
wandb.log({"acc/test": acc})