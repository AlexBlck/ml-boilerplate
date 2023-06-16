import os
import shutil
from os.path import join

import wandb
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .transforms import MyAwesomeTransforms

__all__ = ["MyAwesomeDataModule", "MyAwesomeDataset"]


class MyAwesomeDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasets = {}
        self.samplers = {}
        self.data_root = None

    def prepare_data(self):
        api = wandb.Api()
        artifact = api.artifact(self.hparams.artifact)
        # Create artifact directory
        artifact_dir = join(self.hparams.data_root, artifact.name)

        # If alias is 'latest', delete folder contents. Wandb doesn't overwrite
        if "latest" in artifact.aliases:
            shutil.rmtree(artifact_dir)
        os.makedirs(artifact_dir, exist_ok=True)

        # Download artifact
        self.data_root = artifact.download(root=artifact_dir)

    def setup(self, stage):
        splits = []
        if stage == "fit":
            splits = ["train", "val"]
        elif stage == "test":
            splits = ["test"]
        elif stage == "validate":
            splits = ["val"]
        elif stage == "predict":
            splits = ["predict"]

        # Initialize datasets
        self.datasets = {
            split: MyAwesomeDataset(
                root=self.data_root,
                split=split,
                transforms=MyAwesomeTransforms(split=split, size=self.hparams.img_size),
            )
            for split in splits
        }

        # Samplers
        self.samplers = {split: None for split in splits}

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            sampler=self.samplers["train"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            sampler=self.samplers["val"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            sampler=self.samplers["test"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            sampler=self.samplers["predict"],
        )


class MyAwesomeDataset(Dataset):
    def __init__(self, root, split, transforms):
        self.root = root
        self.split = split
        self.transforms = transforms
        self._load_data()

    def _load_data(self):
        raise NotImplementedError("Dataset loading not implemented!")

    def __getitem__(self, idx):
        sample = {}
        # TODO: Implement your data loading logic here
        # sample["image"] = ...
        sample = self.transforms(sample, split=self.split)
        return sample

    def __len__(self):
        raise NotImplementedError("Dataset length not implemented!")
