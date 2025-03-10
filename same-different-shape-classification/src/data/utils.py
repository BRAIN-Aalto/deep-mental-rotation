import autoroot

from typing import Any, Iterator, Callable
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler, default_collate
from torchvision import io, transforms, utils

from shaperenderer.utils import sample_on_sphere
from shaperenderer.geometry import (
    Plane,
    ShapeString,
    MetzlerShape
)
from shaperenderer.renderer import (
    Object3D,
    Camera,
    Renderer
)


# type alias for a shape pair sample 
ShapePairSample = tuple[ShapeString, Plane, list[float], list[float]]


class SameDifferentShapeDataset(Dataset):
    """
    """
    def __init__(
        self,
        root: str,
        data_file: str,
        transforms: transforms.Compose | None = None
    ) -> None:
        with open(
            os.path.join(root, data_file),
            "r",
            encoding="utf-8"
        ) as reader:
            self.dataset: list[dict[str, Any]] = json.load(reader)

        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        image_1 = io.read_image(os.path.join(self.root, self.dataset[idx]["image_1"]["path"]))[:3, ...] / 255.
        image_2 = io.read_image(os.path.join(self.root, self.dataset[idx]["image_2"]["path"]))[:3, ...] / 255.
        target = torch.tensor([self.dataset[idx]["label"]], dtype=torch.float32)

        if self.transforms:
            if isinstance(self.transforms, transforms.v2.Compose):
                image_1 = self.transforms(image_1)
                image_2 = self.transforms(image_2)

            elif isinstance(self.transforms, list):
                for transform in self.transforms:
                    if isinstance(transform, transforms.v2.RandomVerticalFlip) or isinstance(transform, transforms.v2.RandomRotation):
                        images = transform(torch.concat([image_1, image_2]))

                        image_1 = images[:3, ...]
                        image_2 = images[3:, ...]
                    else:
                        image_1 = transform(image_1)
                        image_2 = transform(image_2)

            # elif isinstance(self.transform, transforms.v2.RandomVerticalFlip):
            #     images = self.transform(torch.concat([image_1, image_2]))

            #     image_1 = images[:3, ...]
            #     image_2 = images[3:, ...]

            #     # + torchvision.transforms.v2.Normalize
            #     normalizer = transforms.v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            #     image_1 = normalizer(image_1)
            #     image_2 = normalizer(image_2)


        return {
            "image_1": image_1,
            "image_2": image_2,
            "label": target
        }


    def __len__(self) -> int:
        return len(self.dataset)
    
    
    def show_batch(self, batch_size: int = 8) -> None:
        """
        """
        indices = np.random.default_rng(seed=12345).integers(0, self.__len__(), size=batch_size)

        fig = plt.figure(figsize=(16, 8), layout="tight")

        for idx, sample_idx in enumerate(indices):
            sample = self.__getitem__(sample_idx)

            img1, img2, label = sample["image_1"], sample["image_2"], sample["label"]

            sample = utils.make_grid([img1, img2], nrow=2, padding=3).permute(1, 2, 0).numpy()

            ax = fig.add_subplot((batch_size // 4) + 1, 4, idx + 1)

            ax.imshow(sample)
            ax.set_title(f"Class {int(label.item())}: {'same' if label else 'different'}")
            ax.axis("off")



class SameDifferentShapeIterableDataset(IterableDataset):
    """
    """
    def __init__(
        self,
        root: str,
        data_file: str,
        seed: int | None = None
    ) -> None:
        with open(
            os.path.join(root, data_file),
            "r",
            encoding="utf-8"
        ) as reader:
            shapes = reader.read().splitlines()
            self.dataset = list(map(ShapeString, shapes))

        self.rng = np.random.default_rng(seed=seed)

    
    def __iter__(self) -> Iterator[tuple[ShapeString, list[float], list[float]]]:
        while True:
            shape_idx = self.rng.integers(low=0, high=len(self.dataset))

            thetas, phies = sample_on_sphere(low=0.05, high=0.95, size=(1, 2), rng=self.rng)
            thetas = np.repeat(thetas, 2)

            yield (self.dataset[shape_idx], thetas.tolist(), phies.tolist())


class ShapePairBatchSampler(Sampler):
    """
    """
    def __init__(
            self,
            dataset: SameDifferentShapeIterableDataset,
            batch_size: int, # must be devisible by 2
            shuffle: bool = False
    ) -> None:
        if batch_size % 2:
            raise ValueError(f"batch_size must be devisible by 2.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __iter__(self) -> Iterator[list[ShapePairSample]]:
        dataset_iter = iter(self.dataset)

        batch = [] # balanced batch of same and different shape pairs
        # sample the positive examples, 50% of the batch
        for _ in range(int(self.batch_size / 2)):
            shape, thetas, phies = next(dataset_iter)

            batch.append(
                (shape, None, thetas, phies) # no reflection plane
            )

        # now sample the negative examples, the rest 50% of the batch
        for _ in range(int(self.batch_size / 2)):
            shape, thetas, phies = next(dataset_iter)
            plane_idx = self.dataset.rng.integers(low=0, high=3)

            batch.append(
                (shape, Plane(plane_idx), thetas, phies)
            )

        # shuffle samples in the batch
        if self.shuffle:
            shuffled = self.dataset.rng.permutation(self.batch_size)
            yield [batch[idx] for idx in shuffled]
        else:
            yield batch


        
def custom_collate_fn(batch: list[ShapePairSample]) -> dict[str, torch.Tensor]:
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

    for sample in batch:
        # step 1: unpack sample
        shape, plane, thetas, phies = sample

        # step 2: generate 3D shape pair based on the shape string and reflection plane
        original = Object3D(
            shape=MetzlerShape(shape),
            **shape_params
        )

        if plane is not None:
            reflected = Object3D(
                shape=MetzlerShape(shape.reflect(over=plane)),
                **shape_params
            )
            objects = (original, reflected)
            label = 0. # shapes are different

        else:
            objects = (original, original)
            label = 1. # shapes are same

        
        sample_dict = dict.fromkeys(("image_1", "image_2", "label"))
        sample_dict["label"] = np.array([label])

        # step 3: render 3-channel images of the shape pair
        for img_idx, (obj, theta, phi) in enumerate(zip(objects, thetas, phies)):
            # position the camera
            camera.setSphericalPosition(
                r=25,
                theta=theta,
                phi=phi
            )
            # render image with the shape viewed from given theta and phi angles
            renderer.render(obj, camera)
            # convert the rendered image into a numpy array
            sample_dict[f"image_{img_idx + 1}"] = renderer.save_figure_to_numpy(color_channel_to_beginning=True)


        # step 4: add sample to a batch of data
        batch_data.append(sample_dict)

    # step 5: collate samples into a batch

    # default_collate returns a dictionary with the same set of keys as in sample_dict,
    # but batched tensors as values: 
    # {
    #   "image_1": torch.tensor([]), # (batch_size, 3, image_height, image_width)
    #   "image_2": torch.tensor([]), # (batch_size, 3, image_height, image_width)
    #   "label": torch.tensor([]),   # (batch_size, 1)
    # }
    return default_collate(batch_data)


class SameDifferentShapeDataLoader:
    """
    """
    def __init__(
            self,
            dataset: SameDifferentShapeIterableDataset,
            batch_sampler: ShapePairBatchSampler,
            collate_fn: Callable[[list[ShapePairSample]], dict[str, torch.Tensor]],
            steps: int = 1
    ) -> None:
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.steps = steps


    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for _ in range(self.steps):
            for batch in self.batch_sampler:
                yield self.collate_fn(batch)


    def __len__(self) -> int:
        return self.steps


    