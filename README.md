# Deep Learning Boilerplate

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

This is a boilerplate for deep learning projects.
It is based on the [PyTorch](https://pytorch.org/)
framework and [PyTorch Lightning](https://www.pytorchlightning.ai/),
utilizing [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html) interface.
It is designed to be used with [Weights & Biases](https://wandb.ai/site) for experiment tracking and visualization,
dataset and model versioning with [artifacts](https://docs.wandb.ai/guides/artifacts) and parameter search
with [sweeps](https://docs.wandb.ai/guides/sweeps).

## Structure

The project is structured as follows:

```
├── configs
│   ├── data.yaml
│   ├── model.yaml
│   └── ...
├── src
│   ├── data
│   │   ├── dataset.py
│   │   └── ...
│   ├── models
│   │   ├── model.py
│   │   └── ...
│   └── main.py
├── run.sh
├── sweep.sh
└── ...
```

There are two wayt to run the code:

1. `./run.sh $command` - runs the code with one of `fit, test, validate, predict`
1. `./sweep.sh` - [sweeps](https://docs.wandb.ai/guides/sweeps) over the parameters specified in `sweep.yaml`

## After cloning

Everything that (probably) needs to be renamed has the wrod `Awesome` in the name. After you think you're done with the setup, you can perform a global search for `awesome` (case-insensitive) to make sure you haven't missed anything.

### Dataset

1. In `src/data/dataset.py` populate `MyAwesomeDataset` with your dataset logic.
1. Unless you need a custom data sampler, you can leave `MyAwesomeDataModule` as is. Otherwise, in `setup()` replace `self.samplers = {split: None for split in splits}` with your sampler.
1. In `src/data/transforms.py` populate `MyAwesomeTransforms` with your dataset transforms.
1. _upload dataset reference as artifact to wandb_
1. In `configs/data.yaml` point `artifact` to wandb link and `data_root` to the local data directory.
   Then define any other dataset parameters you wish to control, such as batch size or image size.

### Model

1. In `src/model/model.py` populate `MyAwesomeModel` with your model logic.
1. In `configs/model.yaml` define any model parameters you wish to control, including losses.
1. In `main.py` you may want to enable automatic checkpoint upload to s3 after fit, as well as configure artifact metadata for `model_artifact`.
1. In `src/model/metrics.py` instantiate any metrics you want to use in `MyAwesomeMetricsWrapper` constructor and assign them per split. You can either use existing [TorchMetrics](https://torchmetrics.readthedocs.io/) or implement your own in `MyAwesomeMetric`.
1. In `src/model/logging.py` populate `SummaryImg` with any postprocessing logic you want to apply to the model output before logging it to wandb.

### Configs

1. Configure optimizers and schedulers in `configs/optimizer.yaml`.
1. Configure training parameters, checkpoint (and other) callbacks and wandb logging in `configs/trainer.yaml`

### Sweeps

If you wish to perform parameter search, you can define the parameters in `sweep.yaml` and run `./sweep.sh`.
