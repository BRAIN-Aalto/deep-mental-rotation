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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from torchmetrics.regression import MeanSquaredError
from torchvision import transforms
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
        rotated = model.mirror_and_rotate(repr, rotation_params)
    else:
        rotated = model.rotate_from_source_to_target(
            repr,
            **rotation_params
        )

    rendered = model.render(rotated)

    return rendered.detach()


def run_bayesian_optimization(seed: int = 12345):
    """
    """

    def log_step(plot_acq: bool = True):
        """
        """
        figure = plt.figure(figsize=(12, 5))
        grid = GridSpec(2, 5, figure=figure, hspace=0.3)


        def plot_GP(optim, y_best, ax):
            """
            """
            ax.set_xlim(-180 - 5, 180 + 5)
            ax.set_xticks(np.linspace(-180, 180, 9))

            X_test = np.linspace(-180, 180, 200).reshape(-1, 1)

            mean, std = optim.surrogate.predict(X_test, optim.history.X, optim.history.Y, return_std=True)
            uncertainty = 1.96 * std # 95% confidence interval
            samples = optim.surrogate.sample(
                X_test,
                X_train=optim.history.X,
                Y_train=optim.history.Y,
                n_samples=5
            )
            
            # plot mean function
            ax.plot(X_test.ravel(), mean.ravel())
            # plot uncertainty area
            ax.fill_between(X_test.ravel(), (mean + uncertainty).ravel(), (mean - uncertainty).ravel(), alpha=0.1)

            # plot sampled point
            if optim is boptim: ax.axvline(rotate_by, c="green", linestyle="--")

            # plot training points
            if not optim.history.empty:
                ax.plot(
                    optim.history.X.ravel(),
                    optim.history.Y.ravel(), 
                    "kX",
                    markersize=5,
                    markeredgewidth=0.3
                )

                # plot the current best loss value
                ax.axhline(y_best, c="black", linestyle="--")

                for sample in samples:
                    ax.plot(X_test, sample, lw=0.5, ls='--')


        ### Top plot: surrogate function (GP) ###
        ax11 = figure.add_subplot(grid[0, :2])

        ax12 = figure.add_subplot(grid[0, 2:4], sharey=ax11)
        ax12.tick_params(axis="y", labelleft=False)

        for optim, ax in zip([sameoptim, diffoptim], [ax11, ax12]):
            plot_GP(optim=optim, y_best=loss_best, ax=ax)




        ## Bottom plot: acquisition function ###
        ax21 = figure.add_subplot(grid[1, :2])

        ax22 = figure.add_subplot(grid[1, 2:4], sharey=ax21)
        ax22.tick_params(axis="y", labelleft=False)

        if plot_acq:

            X_tries = boptim.space.sample(mode="fixed")

            if acqs[0] is not None:

                for optim, acq, ax in zip([sameoptim, diffoptim], acqs, [ax21, ax22]):
                    ax.set_xlim(-180 - 5, 180 + 5)
                    ax.set_xticks(np.linspace(-180, 180, 9))

                    ax.plot(
                        X_tries,
                        acq.ravel(),
                        c="red",
                    )

                    if optim is boptim: ax.axvline(rotate_by, c="green", linestyle="--")


        # Top-left plot: target image ###
        ax3 = figure.add_subplot(grid[0, -1])
        ax3.axis("off")
        ax3.set_title(f"phi = {phi_target:.2f}", fontsize=10)

        ax3.imshow(target.squeeze().cpu().permute(1, 2 ,0))


        ## Bottom-left plot: rotated image ###
        ax4 = figure.add_subplot(grid[-1, -1])
        ax4.axis("off")


        phi_estimated = phi_source + torch.tensor(rotate_by).to(device)
        rendered = transform_and_render(
            repr=repr,
            model=model,
            rotation_params={
                "azimuth_source": phi_source,
                "elevation_source": theta_source,
                "azimuth_target": phi_estimated,
                "elevation_target": theta_source
            },
            mirror=boptim.objective.keywords["mirror"]
        )
        ax4.imshow(rendered.squeeze().cpu().permute(1, 2 ,0))
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
    
    
    # load params to set up the prior distribution
    with open(
        "prior.json",
        "r",
        encoding="utf-8"
    ) as writer:
        params = json.load(writer)
    
    points, mean, cov = np.array(params["points"]), params["mean"], np.array(params["cov"])

    # same
    sameoptim = BayesianOptimizer(
        obj_func=partial(objective_function, mirror=False),
        search_space=LinearGridSpace(start=-180, end=180, size=181, random_state=None),
        surrogate=GaussianProcessRegressor(
            mean=MeanFunction(mean=mean),
            kernel=CustomKernel(cov=cov, points=points),
            random_state=None
        ),
        acquisition=AcquisitionFunction(eps=0.0001, kind="ei")
    )

    # different
    diffoptim = BayesianOptimizer(
        obj_func=partial(objective_function, mirror=True),
        search_space=LinearGridSpace(start=-180, end=180, size=181, random_state=None),
        surrogate=GaussianProcessRegressor(
            mean=MeanFunction(mean=mean),
            kernel=CustomKernel(cov=cov, points=points),
            random_state=None
        ),
        acquisition=AcquisitionFunction(eps=0.0001, kind="ei")
    )

    # stopping_criterion = SearchStoppingCriterion(kind="sampling", threshold=args.stop_threshold)

    rotation_cost = lambda rotate_by: 1 - 0.75 * abs(rotate_by) / 180 # optional

    def step():
        """
        """
        res = defaultdict(list)

        optims = [sameoptim, diffoptim]
        current_best = max(sameoptim.history.maximum["Y"], diffoptim.history.maximum["Y"])

        for optim in optims:
            x_next, y_next, acq = optim.step(
                current_best,
                cost_func=rotation_cost,
                return_acq=True
            )
            res["x"]   += [x_next]
            res["y"]   += [y_next]
            res["acq"] += [acq]

        try:
            best = np.nanargmax(np.nanmax(res["acq"], axis=1, keepdims=True))
            return res["x"][best], res["y"][best], current_best, res["acq"], optims[best]
        
        except Exception:
            return res["x"][0], res["y"][0], current_best, res["acq"], optims[0] # always measure similarity loss first
    


    for i in range(1, args.iters+1):
        rotate_by, loss_value, loss_best, acqs, boptim = step()

        # log_step()
        
        boptim.log({"x": rotate_by, "y": loss_value}, step=i)

        ### stopping criterion 1: loss function sampling
        # if stopping_criterion(
        #     optim=[sameoptim, diffoptim],
        #     y_best=loss_best,
        #     X=np.linspace(-180, 180, 200).reshape(-1, 1)
        # ): break

        ### stopping criterion 2: global loss threshold for match / mismatch
        if -loss_value < args.match_threshold:
            match = 1 if boptim is sameoptim else 0
            break

    else:
        match = None

    log_step()

    return {
        "phi_estimated": phi_source.item() + rotate_by,
        "mse_score": -loss_value,
        "prediction": match,
        "steps": i,
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
        mse = {res["mse_score"]:.5f}, \
        phi (correct) = {phi_target:.2f}, \
        phi (estimated) = {res["phi_estimated"]:.2f}.\n\n'''
    )
    break


data = pd.DataFrame(table)

wandbtable = wandb.Table(data=data)
wandb.log({"result": wandbtable})

acc = 100 * np.sum(data["prediction"] == data["label"]) / len(data)
logger.info(f"Accuracy: {acc:.2f}%")
wandb.log({"acc/test": acc})