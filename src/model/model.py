import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from .logging import SummaryImg
from .metrics import MyAwesomeMetricsWrapper

__all__ = ["MyAwesomeModel"]


class MyAwesomeModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.metrics = MyAwesomeMetricsWrapper(self.hparams.n_classes)
        self.summary_img = SummaryImg(self.hparams.n_classes, self.logger)

        # TODO: Define model layers here
        # self.out = nn.Conv2d(2, self.hparams.n_classes, kernel_size=1)

        # TODO: Change input shape to match your data
        expected_input_shape = (self.hparams.n_channels, 512, 512)
        summary(self, expected_input_shape, device=str(self.device))

    def forward(self, x):
        raise NotImplementedError

    def _fit_step(self, batch, batch_idx, stage):
        # Read input data
        x = batch["image"][:, : self.hparams.n_channels, :, :]
        y = batch["mask"]

        # TODO: Do your logic and forward pass here
        # y_hat = self.forward(x)

        # Calculate loss
        loss = self.hparams.loss(y_hat, y)

        # Log loss
        self.log(f"{stage}_loss", loss, on_step=stage == "train")

        # Update metrics
        self.metrics.update(preds, target)

        # Log images (less frequently than every step)
        if batch_idx % self.hparams.log_freq == 0:
            self.summary_img(x, y, y_hat, stage=stage)

        return loss

    def on_train_start(self):
        pass

    def training_step(self, batch, batch_idx):
        # Calculate loss
        loss = self._fit_step(batch, batch_idx, stage="train")

        # TODO: put any training only logic here

        return {"loss": loss}

    def on_fit_epoch_end(self, stage):
        # Log Metrics
        metrics = self.metrics.compute()
        # Log all of yor metrics
        # fig = metrics["confusion_matrix"]
        # self.logger.experiment.log({"confusion": fig})
        self.metrics.reset()

    def on_train_epoch_end(self):
        self.on_fit_epoch_end("train")
        # TODO: put any training only logic here

    def on_validation_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        # Calculate loss
        loss = self._fit_step(batch, batch_idx, stage="val")

        # TODO: put any validation only logic here

        return {"loss": loss}

    def on_validation_epoch_end(self):
        self.on_fit_epoch_end("val")
        # TODO: put any validation only logic here

    def test_step(self, batch, batch_idx):
        # Read input data
        x = batch["image"]
        y = batch["mask"]

        y_hat = self.forward(x)

        self.metrics.update(preds, target)

    def on_test_end(self):
        metrics = self.metrics.compute(stage="test")
        for metric, value in metrics.items():
            if type(value) == Image:
                self.logger.log_image(f"test_{metric}", [value])
            self.log(f"test_{metric}", value)
