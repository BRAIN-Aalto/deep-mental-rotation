import logging
import os
import json
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from shaperenderer.utils import sample_interval_union, InPlaneRandomRotation
from shaperenderer.geometry import (
    Plane,
    ShapeString,
    MetzlerShape
)
from shaperenderer.renderer import (
    Camera,
    Renderer,
    Object3D
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg : DictConfig) -> None:
    """
    """
    OmegaConf.save(cfg, os.path.join(cfg.paths.output_dir, "config.yaml"))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)


    rng = np.random.default_rng(seed=cfg.seed)
    

    # read shape skeletons from the file
    with open(
        cfg.data.path_to_shapes,
        "r",
        encoding="utf-8"
    ) as reader:
        shapes = reader.read().splitlines()
    shapes = list(map(ShapeString, shapes))


    # configure 3D Shape Renderer
    camera = Camera()

    renderer = Renderer(
        imgsize=(
            cfg.shaperenderer.renderer.image_size,
            cfg.shaperenderer.renderer.image_size
        ),
        dpi=cfg.shaperenderer.renderer.dpi,
        bgcolor=cfg.shaperenderer.renderer.background,
        format=cfg.shaperenderer.renderer.format
    )

    object_params = {
        "facecolor": cfg.shaperenderer.object.facecolor,
        "edgecolor": cfg.shaperenderer.object.edgecolor,
        "edgewidth": cfg.shaperenderer.object.edgewidth
    }

    samples = rng.choice(len(shapes), size=cfg.data.num_samples)
    labels = (np.arange(cfg.data.num_samples) < cfg.data.num_samples // cfg.data.num_classes).astype(int).tolist()


    PI = 180. # (in degrees)
    OFFSET = 10. # (in degrees)

    if cfg.rotation == "picture":
        theta_space = [
            (-PI/2 + OFFSET, 0*PI - OFFSET),
            (0*PI + OFFSET, PI/2 - OFFSET)
        ]
    elif cfg.rotation == "depth":
        theta_space = [
            (-PI/4, 0*PI - OFFSET),
            (0*PI + OFFSET, PI/4)
        ]

    thetas = sample_interval_union(theta_space, size=cfg.data.num_samples, rng=rng)

    phi_space = [
        (0*PI + OFFSET, PI/2 - OFFSET),
        (PI/2 + OFFSET, PI - OFFSET),
        (PI + OFFSET, 3*PI/2 - OFFSET),
        (3*PI/2 + OFFSET, 2*PI - OFFSET)
    ]
    phies = sample_interval_union(phi_space, size=cfg.data.num_samples, rng=rng)

    if cfg.rotation == "depth":
        low = cfg.get("min_angle", 0.)
        high = cfg.get("max_angle", 360.)
        deltas = rng.uniform(low=low, high=high, size=cfg.data.num_samples)


    data = []
    for sample_idx, (shape_idx, label, theta, phi) in enumerate(zip(samples, labels, thetas, phies)):
        if label == 1: # same pair
            pair = [
                # shape 1
                Object3D(
                    shape=MetzlerShape(shapes[shape_idx]),
                    **object_params
                ),
                # shape 2
                Object3D(
                    shape=MetzlerShape(shapes[shape_idx]),
                    **object_params
                )
            ]
        elif label == 0: # different pair
            pair = [
                # shape 1 (normal)
                Object3D(
                    shape=MetzlerShape(shapes[shape_idx]),
                    **object_params
                ),
                # shape 2 (mirror)
                Object3D(
                    shape=MetzlerShape(shapes[shape_idx].reflect(over=Plane(2))), # mirror canonical view of shape 1 around Z-axis
                    **object_params
                )
            ]

        pair_params = dict.fromkeys(("image_1", "image_2", "label"))
        pair_params["label"] = label

        for i, shape3D in enumerate(pair):
            if cfg.rotation == "picture":
                camera.setSphericalPosition(
                    r=cfg.shaperenderer.camera.distance,
                    theta=theta,
                    phi=phi
                )
                renderer.render(
                    shape3D,
                    camera,
                    transforms=[InPlaneRandomRotation(degrees=(0., 360.))] if i else None
                )
            elif cfg.rotation == "depth":
                camera.setSphericalPosition(
                    r=cfg.shaperenderer.camera.distance,
                    theta=theta,
                    phi=phi + deltas[sample_idx] if i else phi
                )
                renderer.render(
                    shape3D,
                    camera
                )

            image_path = Path(cfg.paths.output_dir) / "images" / f"{sample_idx:04d}_{i+1}.png"
            renderer.save_figure_to_file(image_path, verbose=False)
            

            # track the dataset parameters in the dictionary
            pair_params[f"image_{i+1}"] = dict(
                path=str(image_path.relative_to(Path.cwd())),
                azimuth=phi + deltas[sample_idx] if i else phi,
                elevation=theta
            )
        
        data.append(pair_params)
        

    with open(
        Path(cfg.paths.output_dir) / "data_params.json",
        "w",
        encoding="utf-8"
    ) as writer:
        json.dump(data, writer, indent=4, default=str)

    logger.info(f"Image files and data_params.json saved to {cfg.paths.output_dir}.")


if __name__ == "__main__":
    main()