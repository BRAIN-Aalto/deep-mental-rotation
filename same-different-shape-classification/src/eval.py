import autoroot

import os
import time
import logging

import torch
from torch import nn
from lightning import LightningDataModule
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src import utils
from src.trainer import Trainer


@hydra.main(version_base=None, config_path=os.path.join(os.getenv("PROJECT_ROOT"), "configs"), config_name="evaluate")
def evaluate(cfg: DictConfig) -> None:
    """
    """
    OmegaConf.save(cfg, os.path.join(cfg.paths.output_dir, "config.yaml"))
                   
    logger = utils.setup_logger(__name__)

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        project="same-different-shape-classifier",
        name=f"run-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
        dir=os.path.join(cfg.paths.output_dir, "wandb"),
        config=dict(cfg),
        job_type=cfg.job_name
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    utils.seed_everything(cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = hydra.utils.instantiate(cfg.model).to(device)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup(stage="test")

    criterion = nn.BCELoss()

    trainer = Trainer(
        root_dir=cfg.paths.output_dir,
        device=device,
        logger=logger,
        max_epochs=cfg.trainer.max_epochs,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
    )
    # Evaluate on the training set (optional)
    train_loss, train_acc = trainer.eval(
        model=model,
        test_dataloader=datamodule.test_dataloader(dataset="train"),
        ckpt_path=cfg.ckpt_path,
        criterion=criterion
    )
    logger.info(
        f"""[Train] Loss: {train_loss:.5f}, Accuracy: {100 * train_acc:.2f}%."""
    )
    wandb.log({"loss/train": train_loss})
    wandb.log({"acc/train": 100 * train_acc})


    # Evaluate on the test set
    test_loss, test_acc = trainer.eval(
        model=model,
        test_dataloader=datamodule.test_dataloader(dataset="test"),
        ckpt_path=cfg.ckpt_path,
        criterion=criterion
    )
    logger.info(
        f"""[Test] Loss: {test_loss:.5f}, Accuracy: {100 * test_acc:.2f}%."""
    )
    wandb.log({"loss/test": test_loss})
    wandb.log({"acc/test": 100 * test_acc})


if __name__ == "__main__":
    evaluate()