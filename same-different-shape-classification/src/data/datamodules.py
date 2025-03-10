import os

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.utils import (
    SameDifferentShapeDataset,
    SameDifferentShapeIterableDataset,
)

"""
The LightningDataModule is a convenient way to manage data in PyTorch Lightning.
It encapsulates training, validation, testing, and prediction dataloaders,
as well as any necessary steps for data processing, downloads, and transformations.

The LightningDataModule implements 6 key methods:
    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process (i.e. tokenize), save to disk, etc...

    def setup(self, stage):
        # things to do on every process in DDP
        # load data, set variables, build vocabulary, split data, create datasets, apply data transforms (defined explicitly), etc...

    def train_dataloader(self):
        # return train dataloader

    def val_dataloader(self):
        # return validation dataloader

    def test_dataloader(self):
        # return test dataloader

    def teardown(self):
        # called on every process in DDP
        # clean up after fit, validate, test or predict

[https://lightning.ai/docs/pytorch/stable/data/datamodule.html?highlight=datamodule]
"""

class SameDifferentShapeDataModule(pl.LightningDataModule):
    """
    The way to use the LightningDataModule in plain Pytorch, i.e. withough Lightning

    dm = SameDifferentShapeDataModule()

    # optional
    dm.prepare_data()

    # at the training time
    dm.setup(stage="fit")

    for batch in dm.train_dataloder():
        ...

    for batch in dm.val_dataloder():
        ...

    # optional
    dm.teardown(stage="fit")

    # at the testing time
    dm.setup(stage="test")

    for batch in dm.test_dataloader():
        ...

    # optinal
    dm.teardown(stage="test")
    """
    def __init__(
        self,
        root: str,
        data_dir: str,
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        transforms: dict[str, transforms.Compose] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 12345
            
    ) -> None:
        super().__init__()
        self.root = root
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.transforms = transforms or {"train": None, "inference": None}
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = SameDifferentShapeDataset(
                root=self.root,
                data_file=os.path.join(self.data_dir, "train/data_params.json"),
                transforms=self.transforms["train"]
            )

            if os.path.exists(os.path.join(self.root, self.data_dir, "val")):
                self.val_dataset = SameDifferentShapeDataset(
                    root=self.root,
                    data_file=os.path.join(self.data_dir, "val/data_params.json"),
                    transforms=self.transforms["train"]
            )
            else:
                self.val_dataset = None


        if stage == "test":
            self.train_dataset = SameDifferentShapeDataset(
                root=self.root,
                data_file=os.path.join(self.data_dir, "train/data_params.json"),
                transforms=self.transforms["inference"]
            )

            self.test_dataset = SameDifferentShapeDataset(
                root=self.root,
                data_file=os.path.join(self.data_dir, "test/data_params.json"),
                transforms=self.transforms["inference"]
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=self.rng
        )
    

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=self.rng
        )
    

    def test_dataloader(self, dataset: str = "test") -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset if dataset == "train" else self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=self.rng
        )
    

    @property
    def num_classes(self):
        return 2
    

    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     """Reload datamodule state from a checkpoint"""
    #     return super().load_state_dict(state_dict)


    # def state_dict(self) -> Dict[str, Any]:
    #     """Generate and save datamodel state when saving a chekpoint"""
    #     return super().state_dict()


if __name__ == "__main__":
    _ = SameDifferentShapeDataModule(root=os.getenv("PROJECT_DIR"))