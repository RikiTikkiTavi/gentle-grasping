from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import torch

from gentle_grasp.model.static_sound_aware import StaticSoundAwareGraspSuccessModelModule
from gentle_grasp.data_module import GentleGraspDataModule

import hydra

import mlflow

from contextlib import nullcontext

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: OmegaConf):
    # TODO: Remove reassignments
    mlflow_tracking_uri = cfg.tracking.uri
    experiment_name = cfg.tracking.experiment
    run_name = cfg.tracking.run
    data_path = Path(cfg.dataset_path)
    batch_size = cfg.batch_size
    max_epochs = cfg.max_epochs
    gpu_device = cfg.gpu_device

    torch.set_float32_matmul_precision("high")

    n_folds = cfg.split.get("n_folds", 1)

    print(mlflow_tracking_uri)
    print(cfg.split)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Start the parent run
    with mlflow.start_run(run_name=run_name) as parent_run:
        # Iterate over folds
        for fold_i in range(n_folds):
            # Start the child run if in multi-fold settings
            with (
                mlflow.start_run(
                    run_name=f"fold_{fold_i}_{run_name}",
                    nested=True,
                    parent_run_id=parent_run.info.run_id,
                ) if n_folds > 1 else nullcontext(enter_result=parent_run)
            ) as child_run:

                datamodule = GentleGraspDataModule(
                    data_path=data_path,
                    batch_size=batch_size,
                    num_workers=8,
                    split_strategy=hydra.utils.instantiate(cfg.split, fold=fold_i),
                )

                modelmodule = StaticSoundAwareGraspSuccessModelModule(cfg=cfg)

                # Callbacks
                # Stop if no improvement after 5 epochs
                early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.00)
                lr_monitor = LearningRateMonitor(logging_interval='epoch')

                # Logger
                logger = MLFlowLogger(
                    tracking_uri=mlflow_tracking_uri,
                    run_id=child_run.info.run_id
                )

                # Log hyperparameters
                logger.log_hyperparams(cfg)

                # Trainer
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    callbacks=[
                        early_stop,
                        lr_monitor,
                    ],
                    logger=logger,
                    accelerator="auto",
                    devices=[gpu_device],
                    enable_checkpointing=False,
                )

                trainer.fit(modelmodule, datamodule=datamodule)

                # TODO: Get best/last metrics and log to parent

if __name__ == "__main__":
    main()
