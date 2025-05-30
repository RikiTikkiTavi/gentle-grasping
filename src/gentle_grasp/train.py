from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import torch

from gentle_grasp.model_module import GentleGraspModelModule
from gentle_grasp.data_module import GentleGraspDataModule

import hydra

from torchmetrics import Accuracy

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: dict):
    mlflow_tracking_uri = "https://dagshub.com/RikiTikkiTavi/gentle-grasping.mlflow"
    data_path = Path("/data/horse/ws/s4610340-gentle-grasp/gentle-grasping/data/raw/data_gentle_grasping/gentle_grasping_dataset.pth")
    batch_size = 64

    torch.set_float32_matmul_precision("high")

    datamodule = GentleGraspDataModule(data_path=data_path, batch_size=batch_size, num_workers=2, val_ratio=0.2, cv=cfg.cv)

    modelmodule = GentleGraspModelModule(cfg=cfg)

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min")

    # Logger
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name="gentle_grasping_experiment",
        run_name="action_conditional_model_run",
    )

    # Log hyperparameters
    logger.log_hyperparams(cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[early_stop],
        logger=logger,
        accelerator="auto",
        devices=[6],
        enable_checkpointing=False,
    )

    trainer.fit(modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()