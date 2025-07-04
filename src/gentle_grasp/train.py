from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import torch

from gentle_grasp.model.static_sound_aware import StaticSoundAwareGraspSuccessModelModule
from gentle_grasp.data_module import GentleGraspDataModule

import hydra

import mlflow

from contextlib import nullcontext

OmegaConf.register_new_resolver("sum", sum)

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    transforms = {
        "image": {
            "pre": hydra.utils.instantiate(cfg.transforms.image.pre),
            "augmentations": hydra.utils.instantiate(cfg.transforms.image.augmentations),
            "post": hydra.utils.instantiate(cfg.transforms.image.post),
        },
        "sensor": {
            "pre": hydra.utils.instantiate(cfg.transforms.sensor.pre),
            "augmentations": hydra.utils.instantiate(cfg.transforms.sensor.augmentations),
            "post": hydra.utils.instantiate(cfg.transforms.sensor.post),
        },
        # TODO: Specify sound transforms
        "audio": [hydra.utils.instantiate(i) for i in cfg.transforms.audio]
    }

    torch.set_float32_matmul_precision("high")

    n_folds = cfg.split.get("n_folds", 1)

    print(cfg.tracking.uri)
    print(cfg.split)

    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(experiment_name=cfg.tracking.experiment)

    # Start the parent run
    with mlflow.start_run(run_name=cfg.tracking.run ) as parent_run:
        # Iterate over folds
        for fold_i in range(n_folds):
            # Start the child run if in multi-fold settings
            with (
                mlflow.start_run(
                    run_name=f"fold_{fold_i}_{cfg.tracking.run}",
                    nested=True,
                    parent_run_id=parent_run.info.run_id,
                ) if n_folds > 1 else nullcontext(enter_result=parent_run)
            ) as child_run:

                datamodule = GentleGraspDataModule(
                    data_path=Path(cfg.dataset_path),
                    batch_size=cfg.batch_size,
                    num_workers=8,
                    split_strategy=hydra.utils.instantiate(cfg.split, fold=fold_i),
                    sound_mono=cfg.sound_mono,
                    transforms=transforms,
                )

                modelmodule = StaticSoundAwareGraspSuccessModelModule(cfg=cfg)

                # Callbacks
                # Stop if no improvement after 5 epochs
                early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.00)
                lr_monitor = LearningRateMonitor(logging_interval='epoch')

                # Logger
                logger = MLFlowLogger(
                    tracking_uri=cfg.tracking.uri,
                    run_id=child_run.info.run_id
                )

                # Log hyperparameters
                logger.log_hyperparams(OmegaConf.to_object(cfg)) # type: ignore

                # Trainer
                trainer = pl.Trainer(
                    max_epochs=cfg.max_epochs,
                    callbacks=[
                        early_stop,
                        lr_monitor,
                    ],
                    logger=logger,
                    accelerator="auto",
                    devices=[cfg.gpu_device],
                    enable_checkpointing=False,
                )

                trainer.fit(modelmodule, datamodule=datamodule)

                # TODO: Get best/last metrics and log to parent

if __name__ == "__main__":
    main()
