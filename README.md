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
