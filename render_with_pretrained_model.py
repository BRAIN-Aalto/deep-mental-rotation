import argparse
import wandb
from pathlib import Path
import os
import json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import ToTensor

from models.neural_renderer import load_model
from misc.utils import full_rotation_angle_sequence, sine_squared_angle_sequence
from misc.viz import batch_generate_novel_views, save_img_sequence_as_gif

import warnings; warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "path_to_model",
    type=lambda p: Path(p).absolute(),
    help="path to the trained model"
)
parser.add_argument(
    "path_to_data",
    type=lambda p: Path(p).absolute(),
    help="path to the folder with images and render parameters"
)
parser.add_argument(
    "--save_dir",
    default=Path(__file__).absolute().parent,
    type=lambda p: Path(p).absolute(),
    help="path to the saving directory for data"
)

wandb.login(key=os.getenv("WANDB_KEY"))
run = wandb.init(
    project="metzler-deep-model",
    name="Predictions"
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### load trained model ###
model = load_model(args.path_to_model).to(device)

args.save_dir.mkdir(parents=True, exist_ok=True)
data = []
for shape_folder in args.path_to_data.iterdir():
    image = imageio.imread(shape_folder / "00000.png")
    # convert image to tensor and add batch dimension to it
    img_source = ToTensor()(image[..., :3])
    img_source = img_source.unsqueeze(0).to(device)
    assert img_source.shape[1:] == (3, image.shape[0], image.shape[1]), f"Error: image dimensions are not correct, {img_source.shape[1:]}"

    ### load render parameters ###
    with open(shape_folder / "render_params.json", "r", encoding="utf-8") as reader:
        render_params = json.load(reader)

    azimuth_source = torch.Tensor([render_params["00000"]["azimuth"]]).to(device)
    elevation_source = torch.Tensor([render_params["00000"]["elevation"]]).to(device)

    ### infer scene representation ###
    scene = model.inverse_render(img_source)
    assert scene.shape == (1, 64, 32, 32, 32), f"Error: shape of the scene representation is not correct, {scene.shape}"

    ### generate 360-degree view of the scene ###
    n_frames = 25

    azimuth_shifts = full_rotation_angle_sequence(n_frames).to(device)
    elevation_shifts = sine_squared_angle_sequence(n_frames, -10., 20.).to(device)

    views = batch_generate_novel_views(model, img_source, azimuth_source,
                                    elevation_source, azimuth_shifts,
                                    elevation_shifts)

    save_img_sequence_as_gif(views, args.save_dir / f"{shape_folder.name}.gif")
    data.append(wandb.Video(str(args.save_dir / f"{shape_folder.name}.gif"), caption=f"{shape_folder.name}", fps=20))


wandb.log({f"Novel view synthesis by {args.path_to_model.stem} model": data})
os.rmdir(str(args.save_dir))