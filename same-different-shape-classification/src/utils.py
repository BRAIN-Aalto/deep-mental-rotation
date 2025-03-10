from typing import Optional
import logging
import random

import numpy as np
import torch
from torchvision import utils
import wandb


def seed_everything(seed: int = 12345):
    """
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger(name):
    """
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[list[type]] = None,
    custom_keys_weight_decay: Optional[list[tuple[str, float]]] = None,
):
    """
    Source code: https://github.com/pytorch/vision/blob/d3beb52a00e16c71e821e192bcc592d614a490c0/references/classification/utils.py#L405
    """
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups



def log_batch(batch, title: str):
    images_1, images_2, targets = batch["image_1"], batch["image_2"], batch["label"]

    samples = []
    for idx, (img1, img2, target) in enumerate(zip(images_1, images_2, targets)):
        sample = utils.make_grid([img1, img2], nrow=2, padding=3).permute(1, 2, 0).numpy()

        samples.append(
            wandb.Image(
                sample,
                caption=f"Class {target.item():.0f}: {'same' if target else 'different'}"
            )
        )

        if (idx + 1) % 108 == 0:
            wandb.log({title: samples})
            samples = []

    wandb.log({title: samples})