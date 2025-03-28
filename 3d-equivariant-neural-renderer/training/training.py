import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
import wandb

from models.neural_renderer import NeuralRenderer
from misc.utils import EarlyStopping


class Trainer:
    """Class used to train neural renderers.

    Args:
        device (torch.device):
            Device to train model on.
        model (models.neural_renderer.NeuralRenderer):
            Model to train.
        lr (float):
            Learning rate.
        rendering_loss_type (string):
            One of 'l1', 'l2'.
        ssim_loss_weight (float):
            Weight assigned to SSIM loss.
    """
    def __init__(
        self,
        device: torch.device,
        model: NeuralRenderer,
        lr: float = 2e-4,
        rendering_loss_type: str = "l1",
        ssim_loss_weight: float = 0.05
    ) -> None:
        self.device = device
        self.model = model
        self.lr = lr
        self.rendering_loss_type = rendering_loss_type
        self.ssim_loss_weight = ssim_loss_weight
        self.use_ssim = ssim_loss_weight > 0
        self.register_losses = True # when False doesn't save losses in loss history
        self.multi_gpu = isinstance(self.model, nn.DataParallel) # check if model is multi-gpu

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # Initialize loss functions
        # For rendered images
        if self.rendering_loss_type == "l1":
            self.loss_func = nn.L1Loss()
        elif self.rendering_loss_type == "l2":
            self.loss_func = nn.MSELoss()

        # For SSIM
        if self.ssim_loss_weight:
            # self.ssim_loss_func = SSIM(
            #     data_range=1.0,
            #     size_average=True,
            #     channel=3,
            #     nonnegative_ssim=False
            # )
            self.ssim_loss_func = StructuralSimilarityIndexMeasure(
                data_range=1.0,
                reduction="elementwise_mean",
            )

        # Loss histories
        self.recorded_losses = ["total", "regression", "ssim"]
        self.loss_history = {
            loss_type: []
            for loss_type in self.recorded_losses
        }
        self.epoch_loss_history = {
            loss_type: []
            for loss_type in self.recorded_losses
        }
        self.val_loss_history = {
            loss_type: []
            for loss_type in self.recorded_losses
        }


    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        save_dir: Optional[Path] = None,
        save_freq: int = 1,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        early_stop: bool = False
    ) -> None:
        """Trains a neural renderer model on the given dataloader.

        Args:
            dataloader (torch.utils.DataLoader):
                Dataloader for a misc.dataloaders.SceneRenderDataset instance.
            epochs (int):
                Number of epochs to train for.
            save_dir (string or None):
                If not None, saves model and generated images to directory
                described by save_dir.
                Note that this directory should already exist.
            save_freq (int):
                Frequency with which to save model.
            test_dataloader (torch.utils.DataLoader or None):
                If not None, will test model on this dataset after every epoch.
        """
        if test_dataloader is not None:
            # Extract one batch of validation data
            for batch in test_dataloader: break
        else:
            # Extract one batch of training data
            for batch in dataloader: break

        # Store batch to check how rendered images improve during training
        fixed_batch = batch

        self._log_rendered_batch(fixed_batch)

        if early_stop: early_stopper = EarlyStopping(patience=10)

        for epoch in range(epochs):
            print()
            print(f"Epoch {epoch + 1:03d}")

            self._train_epoch(dataloader)
            
            # Update epoch loss history with mean loss over epoch
            for loss_type in self.recorded_losses:
                self.epoch_loss_history[loss_type].append(
                    sum(self.loss_history[loss_type][-len(dataloader):]) / len(dataloader)
                )

            # Print epoch losses
            print("Mean epoch losses:", end="\t")
            self._print_losses(self.epoch_loss_history)
            wandb.log(
                {
                    "train: regression loss": self.epoch_loss_history["regression"][-1],
                    "train: ssim loss": self.epoch_loss_history["ssim"][-1],
                    "train: total loss": self.epoch_loss_history["total"][-1],
                }
            )

            # Make inference on a fixed batch of images
            # and log the rendered images into W&B
            self._log_rendered_batch(fixed_batch)

            # Optionally save losses and model
            if save_dir is not None:
                # Save losses
                with open(
                    save_dir / "loss_history.json",
                    "w",
                    encoding="utf-8"
                ) as loss_file:
                    json.dump(self.loss_history, loss_file, indent=4)
                # Save epoch losses
                with open(
                    save_dir / "epoch_loss_history.json",
                    "w",
                    encoding="utf-8"
                ) as loss_file:
                    json.dump(self.epoch_loss_history, loss_file, indent=4)

                # Save model
                if (epoch + 1) % save_freq == 0:
                    if self.multi_gpu:
                        self.model.module.save(save_dir / "model.pt")
                    else:
                        self.model.save(save_dir / "model.pt")

            if test_dataloader is not None:                
                regression_loss, ssim_loss, total_loss = \
                    mean_dataset_loss(self, test_dataloader)

                for loss_type, loss_value in zip(
                    self.recorded_losses,
                    (total_loss, regression_loss, ssim_loss)
                ): self.val_loss_history[loss_type].append(loss_value)

                print("Validation epoch losses:", end="\t")
                self._print_losses(self.val_loss_history)

                print(f"Best epoch: {np.argmin(self.val_loss_history['total']) + 1}")

                wandb.log(
                    {
                        "val: regression loss": self.val_loss_history["regression"][-1],
                        "val: ssim loss": self.val_loss_history["ssim"][-1],
                        "val: total loss": self.val_loss_history["total"][-1],
                    }
                )

                if save_dir is not None:
                    # Save validation losses
                    with open(
                        save_dir / "val_loss_history.json",
                        "w",
                        encoding="utf-8"
                    ) as loss_file:
                        json.dump(self.val_loss_history, loss_file, indent=4)
                    # If current validation loss is the lowest,
                    # save model as best model
                    if min(self.val_loss_history["total"]) == total_loss:
                        print("New best model!")

                        if self.multi_gpu:
                            self.model.module.save(save_dir / "best_model.pt")
                        else:
                            self.model.save(save_dir / "best_model.pt")

                # if early_stopper(self.val_loss_history["total"]): break

        # Save model after training is finished
        if save_dir is not None:
            if self.multi_gpu:
                self.model.module.save(save_dir / "model.pt")
            else:
                self.model.save(save_dir / "model.pt")


    def _train_epoch(self, dataloader):
        """Trains model for a single epoch.

        Args:
            dataloader (torch.utils.DataLoader):
                Dataloader for a misc.dataloaders.SceneRenderDataset instance.
        """
        num_iterations = len(dataloader)

        for i, batch in enumerate(dataloader):
            # Train inverse and forward renderer on batch
            self._train_iteration(batch)

            # Print iteration losses
            print(f"{i + 1}/{num_iterations}")
            self._print_losses(self.loss_history)


    def _train_iteration(self, batch):
        """Trains model for a single iteration.

        Args:
            batch (dict):
                Batch of data as returned by a Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
        """
        imgs, rendered, scenes, scenes_rotated = self.model(batch)
        self._optimizer_step(imgs, rendered)


    def _optimizer_step(self, imgs, rendered):
        """Updates weights of neural renderer.

        Args:
            imgs (torch.Tensor):
                Ground truth images. Shape (batch_size, channels, height, width).
            rendered (torch.Tensor):
                Rendered images. Shape (batch_size, channels, height, width).
        """
        self.optimizer.zero_grad()

        loss_regression = self.loss_func(rendered, imgs)

        if self.use_ssim:
            # We want to maximize SSIM, i.e. minimize -SSIM
            loss_ssim = (1. - self.ssim_loss_func(rendered, imgs)).to(loss_regression.device)
            loss_total = loss_regression \
                         + self.ssim_loss_weight * loss_ssim
        else:
            loss_total = loss_regression

        loss_total.backward()

        self.optimizer.step()

        # Record total loss
        if self.register_losses:
            self.loss_history["total"].append(loss_total.item())
            self.loss_history["regression"].append(loss_regression.item())
            if not self.use_ssim: # if SSIM is not used, log down 0
                self.loss_history["ssim"].append(0.)
            else:
                self.loss_history["ssim"].append(loss_ssim.item())


    def _print_losses(self, loss_history):
        """Prints most recent losses."""
        loss_info = []

        for loss_type in self.recorded_losses:
            loss_info += [loss_type, loss_history[loss_type][-1]]

        print("{}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(*loss_info))


    def _render_fixed_batch(self, batch):
        """Reconstructs fixed batch through neural renderer (by inferring
        scenes, rotating them and re-rendering).
        """
        _, rendered, _, _ = self.model(batch)

        return rendered.detach()


    def _log_rendered_batch(self, batch, size: int = 32):
        """
        """
        target = batch["image"].to(self.device)
        # Render images
        rendered = self._render_fixed_batch(batch)

        images_to_log = []
        for idx, (render, gt) in enumerate(zip(rendered, target)):
            img_grid = make_grid(
                [gt, render],
                nrow=2,
                padding=4
            )
            
            images_to_log.append(
                wandb.Image(
                    img_grid.cpu() \
                            .numpy() \
                            .transpose(1, 2, 0),
                )
            )

            if idx + 1 > size: break

        wandb.log({"Rendered shapes (validation set)": images_to_log})


def mean_dataset_loss(trainer, dataloader):
    """Returns the mean loss of a model across a dataloader.

    Args:
        trainer (training.Trainer): Trainer instance containing model to
            evaluate.
        dataloader (torch.utils.DataLoader): Dataloader for a
            misc.dataloaders.SceneRenderDataset instance.
    """
    # No need to calculate gradients during evaluation, so disable gradients to
    # increase performance and reduce memory footprint
    with torch.no_grad():
        # Ensure calculated losses aren't registered as training losses
        trainer.register_losses = False

        regression_loss = 0.
        ssim_loss = 0.
        total_loss = 0.

        for i, batch in enumerate(dataloader):
            imgs, rendered, scenes, scenes_rotated = trainer.model(batch)

            # Update losses
            # Use _loss_func here and not _loss_renderer
            # since we only want regression term
            current_regression_loss = trainer.loss_func(rendered, imgs).item()
            if trainer.use_ssim:
                current_ssim_loss = 1. - trainer.ssim_loss_func(rendered, imgs).item()
            else:
                current_ssim_loss = 0.

            regression_loss += current_regression_loss
            ssim_loss += current_ssim_loss
            total_loss += current_regression_loss \
                            + trainer.ssim_loss_weight * current_ssim_loss

        # Average losses over dataset
        regression_loss /= len(dataloader)
        ssim_loss /= len(dataloader)
        total_loss /= len(dataloader)

        # Reset boolean so we register losses if we continue training
        trainer.register_losses = True

    return regression_loss, ssim_loss, total_loss