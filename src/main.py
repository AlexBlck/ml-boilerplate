import os
from os.path import join
from pathlib import Path

# import boto3  # for uploading to S3
import wandb
from pytorch_lightning.cli import LightningCLI

from .data import MyAwesomeDataModule
from .model import MyAwesomeModel


class MyLightningCLI(LightningCLI):
    wandb_run = None

    def before_fit(self):
        self.wandb_run = self.trainer.logger.experiment
        # Link artifact to wandb logger inside the trainer
        self.wandb_run.use_artifact(self.datamodule.hparams.artifact, type="dataset")

        # Set checkpoint path
        run_id = self.wandb_run.id
        ckpt_path = join(self.model.hparams.ckpt_root, run_id)
        os.makedirs(ckpt_path)
        self.trainer.checkpoint_callback.dirpath = ckpt_path

    def after_fit(self):
        # Upload model to S3
        best_ckpt_path = Path(self.trainer.checkpoint_callback.best_model_path)
        best_ckpt_path = f"file://{best_ckpt_path}"

        # Uncomment to upload to S3
        # file_path = best_ckpt_path.parent
        # file_name = best_ckpt_path.name
        # s3 = boto3.client("s3")
        # s3_bucket_name = "s3-bucket-name"
        # s3_path = f"/path/in/s3/bucket/{file_path}/{file_name}"
        # s3.upload_file(
        #     best_ckpt_path,
        #     s3_bucket_name,
        #     s3_path,
        # )
        # best_ckpt_path = f"s3://{s3_bucket_name}/{s3_path}"

        # Log model artifact to wandb
        model_artifact = wandb.Artifact(
            "my-awesome-model", type="model", metadata={"some-tag": 123}
        )
        model_artifact.add_reference(best_ckpt_path, name="model.ckpt")
        self.wandb_run.log_artifact(model_artifact)

        # Run test
        self.trainer.callbacks = []
        fn_kwargs = {
            "model": self.model,
            "datamodule": self.datamodule,
            "ckpt_path": best_ckpt_path,
        }
        self.trainer.test(**fn_kwargs)


def cli_main():
    MyLightningCLI(
        MyAwesomeModel, MyAwesomeDataModule, save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()
