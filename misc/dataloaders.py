from pathlib import Path
import json
import os
from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Optional,
    Callable,
    Iterator
)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils

from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    IterableDataset,
    #default_collate
)

from misc.metzler_renderer.utils import sample_on_sphere
from misc.metzler_renderer.geometry import Plane, ShapeString, MetzlerShape
from misc.metzler_renderer.renderer import Object3D, Camera, Renderer


class SceneRenderDataset(Dataset):
    """Dataset of rendered scenes and their corresponding camera angles.

    Args:
        path_to_data (string):
            Path to folder containing dataset.
        img_transform (torchvision.transform):
            Transforms to be applied to images.
        allow_odd_num_imgs (bool):
            If True, allows datasets with an odd number of views.
            Such a dataset cannot be used for training,
            since each training iteration requires a *pair* of images.
            Datasets with an odd number of images are used for PSNR calculations.

    Notes:
        - Image paths must be of the form "XXXXX.png" where XXXXX are *five*
        integers indexing the image.
        - We assume there are the same number of rendered images for each scene
        and that this number is even.
        - We assume angles are given in degrees.
    """
    def __init__(
        self,
        root: str,
        path_to_data: str = "chairs-train",
        img_transform: transforms.Compose = None,
        allow_odd_num_imgs: bool = False
    ) -> None:
        self.path_to_data = Path(os.path.join(root, path_to_data))
        self.img_transform = img_transform
        self.allow_odd_num_imgs =  allow_odd_num_imgs
        self.data = []
        # Each folder contains a single scene with different rendering
        # parameters and views
        self.scene_paths = [
            scene_folder
            for scene_folder in self.path_to_data.iterdir()
            if scene_folder.is_dir()
        ]
        # Ensure consistent ordering of scenes
        self.scene_paths.sort()
        self.num_scenes = len(self.scene_paths)
        # Extract number of rendered images per object (which we assume is constant)
        self.num_imgs_per_scene = len(list(self.scene_paths[0].glob("*.png")))
        # If number of images per scene is not even, drop last image
        if self.num_imgs_per_scene % 2 != 0:
            if not self.allow_odd_num_imgs:
                self.num_imgs_per_scene -= 1

        # For each scene, extract its rendered views and render parameters
        for scene_path in self.scene_paths:
            # Name of folder defines scene name
            scene_name = scene_path.name

            # Load render parameters
            with open(
                scene_path / "render_params.json",
                "r",
                encoding="utf-8"
            ) as reader:
                render_params = json.load(reader)

            # Extract path to rendered images of scene
            img_paths = list(scene_path.glob("*.png"))
            # Ensure consistent ordering of images
            img_paths.sort()
            # Ensure number of image paths is even
            img_paths = img_paths[:self.num_imgs_per_scene]

            for img_path in img_paths:
                # Filenames are of the type "<index>.png", so extract this
                # index to match with render parameters.
                img_idx = img_path.stem  # This should be a string
                # Convert render parameters to float32
                img_params = {
                    key: np.float32(value)
                    for key, value in render_params[img_idx].items()
                }
                self.data.append({
                    "scene_name": scene_name,
                    "img_path": img_path,
                    "render_params": img_params
                })


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.data[idx]["img_path"]
        render_params = self.data[idx]["render_params"]

        img = Image.open(img_path)

        # Transform images
        if self.img_transform:
            img = self.img_transform(img)

        # Note some images may contain 4 channels (i.e. RGB + alpha),
        # we only keep RGB channels
        return {
            "scene_name": self.data[idx]["scene_name"],
            "image": img[:3],
            "azimuth": self.data[idx]["render_params"]["azimuth"],
            "elevation": self.data[idx]["render_params"]["elevation"],
        }


class RandomPairSampler(Sampler):
    """Samples random elements in pairs. Dataset is assumed to be composed of a
    number of scenes, each rendered in a number of views. This sampler returns
    rendered image in pairs. I.e. for a batch of size 6, it would return e.g.:

    [object 4 - img 5,
     object 4 - img 12,
     object 6 - img 3,
     object 6 - img 19,
     object 52 - img 10,
     object 52 - img 3]


    Arguments:
        dataset (Dataset):
            Dataset to sample from. This will typically be an
            instance of SceneRenderDataset.
    """

    def __init__(self, dataset: SceneRenderDataset) -> None:
        self.dataset = dataset


    def __iter__(self):
        num_scenes = self.dataset.num_scenes
        num_imgs_per_scene = self.dataset.num_imgs_per_scene

        # Sample num_imgs_per_scene / 2 permutations of the objects
        scene_permutations = [
            torch.randperm(num_scenes)
            for _ in range(num_imgs_per_scene // 2)
        ]
        # For each scene, sample a permutation of its images
        img_permutations = [
            torch.randperm(num_imgs_per_scene)
            for _ in range(num_scenes)
        ]

        data_permutation = []

        for i, scene_permutation in enumerate(scene_permutations):
            for scene_idx in scene_permutation:
                # Extract image permutation for this object
                img_permutation = img_permutations[scene_idx]
                # Add 2 images of this object to data_permutation
                data_permutation.append(
                    scene_idx.item() * num_imgs_per_scene \
                    + img_permutation[2*i].item()
                )
                data_permutation.append(
                    scene_idx.item() * num_imgs_per_scene \
                    + img_permutation[2*i + 1].item()
                )

        return iter(data_permutation)


    def __len__(self):
        return len(self.dataset)


def scene_render_dataset(
    root: str,
    path_to_data: str = "chairs-train",
    img_size: Tuple[int, int, int] = (3, 128, 128),
    crop_size:int = 128,
    allow_odd_num_imgs: bool= False
)-> SceneRenderDataset:
    """Helper function for creating a scene render dataset.

    Args:
        path_to_data (string):
            Path to folder containing dataset.
        img_size (tuple of ints):
            Size of output images.
        crop_size (int):
            Size at which to center crop rendered images.
        allow_odd_num_imgs (int):
            If True, allows datasets with an odd number of views.
            Such a dataset cannot be used for training,
            since each training iteration requires a *pair* of images.
            Datasets with an odd number of images are used for PSNR calculations.
    """
    img_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(img_size[1:]),
        transforms.ToTensor()
    ])

    dataset = SceneRenderDataset(
        root=root,
        path_to_data=path_to_data,
        img_transform=img_transform,
        allow_odd_num_imgs=allow_odd_num_imgs
    )

    return dataset


def scene_render_dataloader(
    root: str,
    path_to_data: str = "chairs-train",
    batch_size: int = 16,
    img_size: Tuple[int, int, int] = (3, 128, 128),
    crop_size: int = 128,
    num_workers: int = 0,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    # generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Dataloader for scene render datasets. Returns scene renders in pairs,
    i.e. 1st and 2nd images are of some scene, 3rd and 4th are of some different
    scene and so on.

    Args:
        path_to_data (string):
            Path to folder containing dataset.
        batch_size (int):
            Batch size for data. Batch size must be even.
        img_size (tuple of ints):
            Size of output images.
        crop_size (int):
            Size at which to center crop rendered images.        
    """
    assert batch_size % 2 == 0, f"Batch size is {batch_size} but must be even"

    dataset = scene_render_dataset(root, path_to_data, img_size, crop_size)

    sampler = RandomPairSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )


def create_batch_from_data_list(
    data_list: List[Dict[str, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """Given a list of datapoints, create a batch.

    Args:
        data_list (list):
            List of items returned by SceneRenderDataset.
    """
    imgs = []
    azimuths = []
    elevations = []

    for data_item in data_list:
        img, render_params = data_item["img"], data_item["render_params"]
        azimuth, elevation = render_params["azimuth"], render_params["elevation"]

        imgs.append(img.unsqueeze(0))
        azimuths.append(torch.Tensor([azimuth]))
        elevations.append(torch.Tensor([elevation]))

    imgs = torch.cat(imgs, dim=0)
    azimuths = torch.cat(azimuths)
    elevations = torch.cat(elevations)

    return (imgs, azimuths, elevations)



ShapePairSample = Tuple[ShapeString, List[float], List[float]]


class ShapePairIterableDataset(IterableDataset):
    """
    """
    def __init__(
            self,
            root: str,
            data_file: str,
            pair: str = "same", # or "different"
            seed: Optional[int] = None
    ) -> None:
        with open(
            os.path.join(root, data_file),
            "r",
            encoding="utf-8"
        ) as reader:
            shapes = reader.read().splitlines()
            self.dataset = list(map(ShapeString, shapes))

        self.pair = pair
        self.rng = np.random.default_rng(seed=seed)

    
    def __iter__(self) -> Iterator[Tuple[List[ShapeString], List[float], List[float]]]:
        while True:
            shape_idx = self.rng.integers(low=0, high=len(self.dataset))
            if self.pair == "same":
                shapes = [
                    self.dataset[shape_idx],
                    self.dataset[shape_idx]
                ]
            elif self.pair == "different":
                shapes = [
                    self.dataset[shape_idx],
                    self.dataset[shape_idx].reflect(over=Plane(2)) # reflect over the picture plane (XY)
                ]
            else:
                raise ValueError(
                    f"""Error: {self.pair} is unknown."""
                )

            thetas, phies = sample_on_sphere(low=0.05, high=0.95, size=(2, 2), rng=self.rng)

            yield (shapes, thetas.tolist(), phies.tolist())


class ShapePairBatchSampler(Sampler):
    """
    """
    def __init__(
            self,
            dataset: ShapePairIterableDataset,
            batch_size: int, # must be devisible by 2
            shuffle: bool = False
    ) -> None:
        if batch_size % 2:
            raise ValueError(f"Error: batch_size must be devisible by 2.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __iter__(self) -> Iterator[List[ShapePairSample]]:
        dataset_iter = iter(self.dataset)

        batch = [next(dataset_iter) for _ in range(self.batch_size)]

        # shuffle samples in the batch
        if self.shuffle:
            shuffled = self.dataset.rng.permutation(self.batch_size)
            yield [batch[idx] for idx in shuffled]
        else:
            yield batch


        
def custom_collate_fn(batch: List[ShapePairSample]) -> Dict[str, torch.Tensor]:
    """
    """
    shape_params = {
        "facecolor": "white",
        "edgecolor": "black",
        "edgewidth": 0.8
    }
    renderer_params = {
        "imgsize": (128, 128),
        "bgcolor": "white",
        "dpi": 100
    }

    camera = Camera()
    renderer = Renderer(**renderer_params)

    batch_data = []
    keys = ("image", "azimuth", "elevation")

    for sample in batch:
        # step 1: unpack sample
        shapes, thetas, phies = sample

        # step 2: render 3-channel images of the shape pair
        for shape, theta, phi in zip(shapes, thetas, phies):
            # generate 3D shape
            object = Object3D(
                shape=MetzlerShape(shape),
                **shape_params
            )
            # position the camera
            camera.setSphericalPosition(
                r=25,
                theta=theta,
                phi=phi
            )
            # render image with the shape viewed from given theta and phi angles
            renderer.render(object, camera)
            # convert the rendered image into a numpy array
            sample_dict = dict.fromkeys(keys)
            sample_dict["image"] = renderer.save_figure_to_numpy(color_channel_to_beginning=True) / 255.
            sample_dict["azimuth"] = phi
            sample_dict["elevation"] = -theta

            # step 3: add sample to a batch of data
            batch_data.append(sample_dict)

    # step 4: collate samples into a batch
    batch = dict.fromkeys(keys)
    for key in keys:
        batch[key] = torch.from_numpy(np.array([sample[key] for sample in batch_data])).to(torch.float32)

    return batch


class ShapePairDataLoader:
    """
    """
    def __init__(
            self,
            dataset: ShapePairIterableDataset,
            batch_sampler: ShapePairBatchSampler,
            collate_fn: Callable[[List[ShapePairSample]], Dict[str, torch.Tensor]],
            steps: int = 1
    ) -> None:
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.steps = steps


    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self.steps):
            for batch in self.batch_sampler:
                yield self.collate_fn(batch)


    def __len__(self) -> int:
        return self.steps
    



class ShapePairMappableDataset(Dataset):
    """
    """
    def __init__(
        self,
        root: str,
        data_file: str,
        transforms: Optional[transforms.Compose] = None
    ) -> None:
        with open(
            os.path.join(root, data_file),
            "r",
            encoding="utf-8"
        ) as reader:
            self.dataset : List[Dict[str, Any]] = json.load(reader)

        self.root = root
        self.transforms = transforms


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_1 = np.asarray(Image.open(os.path.join(self.root, self.dataset[idx]["image_1"]["path"]))).transpose(2, 0, 1)[:3, ...] / 255.
        image_2 = np.asarray(Image.open(os.path.join(self.root, self.dataset[idx]["image_2"]["path"]))).transpose(2, 0, 1)[:3, ...] / 255.

        azimuth = [self.dataset[idx]["image_1"]["azimuth"], self.dataset[idx]["image_2"]["azimuth"]]
        elevation = [self.dataset[idx]["image_1"]["elevation"], self.dataset[idx]["image_2"]["elevation"]]

        if self.transforms:
            image_1 = self.transforms(image_1)
            image_2 = self.transforms(image_2)

        return {
            "image": [image_1, image_2],
            "azimuth": azimuth,
            "elevation": elevation
        }


    def __len__(self) -> int:
        return len(self.dataset)



class SameDifferentShapeDataset(Dataset):
    """
    """
    def __init__(
        self,
        root: str,
        data_file: str,
        transforms: Optional[transforms.Compose] = None
    ) -> None:
        with open(
            os.path.join(root, data_file),
            "r",
            encoding="utf-8"
        ) as reader:
            self.dataset : List[Dict[str, Any]] = json.load(reader)

        self.root = root
        self.transforms = transforms


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_1 = np.asarray(Image.open(os.path.join(self.root, self.dataset[idx]["image_1"]["path"])))[..., :3]
        image_2 = np.asarray(Image.open(os.path.join(self.root, self.dataset[idx]["image_2"]["path"])))[..., :3]

        label = np.array([self.dataset[idx]["label"]], dtype=float)

        if self.transforms:
            image_1 = self.transforms(image_1)
            image_2 = self.transforms(image_2)

        return (image_1, image_2, label)


    def __len__(self) -> int:
        return len(self.dataset)