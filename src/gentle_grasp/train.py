from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from gentle_grasp.model_module import ActionConditionalModel
from gentle_grasp.data_module import GentleGraspDataModule

def main():
    mlflow_tracking_uri = "https://dagshub.com/RikiTikkiTavi/gentle-grasping.mlflow"
    data_path = Path("/data/horse/ws/s4610340-gentle-grasp/gentle-grasping/data/raw/data_gentle_grasping/gentle_grasping_dataset.pth")
    batch_size = 32
    lr = 1e-4

    datamodule = GentleGraspDataModule(data_path=data_path, batch_size=batch_size)

    model = ActionConditionalModel(lr=lr)

    # Callbacks
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Logger
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name="gentle_grasping_experiment",
        run_name="action_conditional_model_run",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        accelerator="auto",
        devices=[6],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
