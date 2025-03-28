import argparse
import logging
from collections import defaultdict
import json
from pathlib import Path
from math import pi

import numpy as np

from shaperenderer.geometry import MetzlerShape
from shaperenderer.renderer import (
    Camera,
    Object3D,
    Renderer
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()
parser.add_argument(
    "path_to_shapes",
    type=lambda p: Path(p).absolute(),
    help="path to the file with shape descriptors"
)
parser.add_argument(
    "--samples",
    default=50,
    type=int,
    help="number of scene views to be sampled"
)
parser.add_argument(
    "--seed",
    default=12345,
    type=int,
    help="random seed"
)
parser.add_argument(
    "--distance_to_camera",
    default=25,
    type=int,
    help="radial distance to the camera from the object"
)
parser.add_argument(
    "--facecolor",
    default="white",
    type=str,
    help="facecolor of the object"
)
parser.add_argument(
    "--edgecolor",
    default="black",
    type=str,
    help="color of object edges"
)
parser.add_argument(
    "--edgewidth",
    default=1.,
    type=float,
    help="width of object edges"
)
parser.add_argument(
    "--image_size",
    default=[128, 128],
    type=int,
    nargs=2,
    help="size of rendered image"
)
parser.add_argument(
    "--background",
    default="white",
    type=str,
    help="background color of rendered image"
)
parser.add_argument(
    "--dpi",
    default=100,
    type=int,
    help="dpi of rendered image"
)
parser.add_argument(
    "--format",
    default="png",
    type=str,
    help="image format"
)
parser.add_argument(
    "--save_dir",
    default=Path(__file__).absolute().parent,
    type=lambda p: Path(p).absolute(),
    help="path to the saving directory for rendered images"
)


args = parser.parse_args()


rng = np.random.default_rng(seed=args.seed)

# read shape descriptors from the file
with open(args.path_to_shapes, "r", encoding="utf-8") as reader:
    shapes = reader.read().splitlines()

# create the camera object
camera = Camera()
# create the renderer object
renderer = Renderer(
    imgsize=tuple(args.image_size),
    dpi=args.dpi,
    bgcolor=args.background,
    format=args.format
    )

# render multiple views for every scene/shape from the file
for shape in shapes:
    logger.info(f"Generating images of shape {shape} ...")

    # make a directory to save different views of the scene
    shape_dir = args.save_dir / Path(f"shape_{shape}")
    shape_dir.mkdir(parents=True, exist_ok=True)

    # create a dictionary to store camera transformations for generating the view
    camera_params = defaultdict(dict)

    # generate 3D object given the shape description
    object3d = Object3D(
        shape=MetzlerShape(shape),
        facecolor=args.facecolor,
        edgecolor=args.edgecolor,
        edgewidth=args.edgewidth
    )

    # sample views uniformly on the full sphere around the object
    u = rng.uniform(0, 1, size=args.samples)
    v = rng.uniform(0, 1, size=args.samples)

    thetas = np.degrees(pi/2 - np.arccos(2*v - 1)) # substitute polar angle with elevation angle
    phies = np.degrees(2*pi*u)

    for index, (theta, phi) in enumerate(zip(thetas, phies)):
        shape_name = f"{index:05d}"
        # position the camera
        camera.setSphericalPosition(r=args.distance_to_camera, theta=theta, phi=phi)
        # render the scene view
        renderer.render(object3d, camera)
        # save the rendered image in the file
        renderer.save_figure_to_file(shape_dir / shape_name, verbose=False)

        # track the camera parameters in the dictionary
        camera_params[shape_name] = {"azimuth": phi, "elevation": -theta}


    # dump the camera parameters for the generated scenes in a json file
    with open(shape_dir / "render_params.json", "w", encoding="utf-8") as writer:
        json.dump(camera_params, writer, indent=4)
    
    logger.info(f"Image files and render_params.json saved to {shape_dir}")
    logger.info("Done!\n\n")


# save script args into a config file
def encode_path(arg: Path) -> str:
    """
    Helper function that gets called for objects that can’t otherwise be serialized.
    It should return a JSON encodable version of the object or raise a TypeError.
    """
    if isinstance(arg, Path):
        return str(arg.name)
    else:
        type_name = arg.__class__.__name__
        raise TypeError(
            f"Object of type '{type_name}' is not JSON serializable!"
        )


with open(args.save_dir / "data-renderer-config.json", "w", encoding="utf-8") as writer:
    json.dump(vars(args), writer, indent=4, default=encode_path)