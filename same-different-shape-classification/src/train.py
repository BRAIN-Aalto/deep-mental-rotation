import autoroot

import os
import time

import torch
from torch import nn
from lightning import LightningDataModule
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src import utils
from src.trainer import Trainer


@hydra.main(version_base=None, config_path=os.path.join(os.getenv("PROJECT_ROOT"), "configs"), config_name="train")
def train(cfg: DictConfig) -> None:
    """
    """
    OmegaConf.save(cfg, os.path.join(cfg.paths.output_dir, "config.yaml"))

    logger = utils.setup_logger(__name__)

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        project="same-different-shape-classifier",
        name=f"run-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
        config=dict(cfg),
        job_type=cfg.job_name
    )

    utils.seed_everything(cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = hydra.utils.instantiate(cfg.model).to(device)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup(stage="fit")

    criterion = hydra.utils.instantiate(cfg.criterion)

    parameters = utils.set_weight_decay(
        model,
        weight_decay=cfg.weight_decay.weight_decay,
        norm_weight_decay=cfg.weight_decay.norm_weight_decay,
    )
    optimizer = hydra.utils.instantiate(cfg.optimizer)(parameters)

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.trainer.max_epochs - cfg.lr_scheduler.lr_warmup_epochs,
        eta_min=0.
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.lr_scheduler.lr_warmup_decay,
        total_iters=cfg.lr_scheduler.lr_warmup_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[cfg.lr_scheduler.lr_warmup_epochs]
    )

    trainer = Trainer(
        root_dir=cfg.paths.output_dir,
        device=device,
        logger=logger,
        max_epochs=cfg.trainer.max_epochs,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
    )
    trainer.fit(
        model=model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ckpt_path=cfg.ckpt_path
    )


if __name__ == "__main__":
    train()
