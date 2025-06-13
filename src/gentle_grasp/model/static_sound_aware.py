from collections import defaultdict
from email.policy import default
import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinarySpecificity,
    ConfusionMatrix,
    BinaryConfusionMatrix
)
from torchvision.models import densenet121
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger
from mlflow import MlflowClient
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile


class AudioCNN(nn.Sequential):

    def __init__(
        self,
        embedding_dim=128, 
        input_channels=1
    ):
        super().__init__(
            # Layer 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Flattening
            nn.Flatten(),

            # Fully Connected Layers
            nn.Linear(128 * 16 * 16, 256),  # Assuming input mel-spectrogram size of 128x128
            nn.ReLU(),
            nn.Dropout(0.5),

            # Embedding Layer
            nn.Linear(256, embedding_dim)
        )


class StaticSoundAwareModel(nn.Module):

    def __init__(
        self,
        dropout_sensory: float = 0.25,
        dropout_action: float = 0.25,
        dropout_final: float = 0.25,
        action_hidden_dim: int = 1024,
        action_embedding_dim: int = 1024,
        sound_embedding_dim: int = 128,
        final_hidden_dim: int = 1024,
        output_dim: int = 1
    ):
        super().__init__()

        # Vision backbones for 3 images
        self.vision_backbone_rgb = self._get_densenet(dropout=dropout_sensory)
        self.vision_backbone_middle = self._get_densenet(dropout=dropout_sensory)
        self.vision_backbone_thumb = self._get_densenet(dropout=dropout_sensory)

        # MLPs for action inputs
        # relpose_action: [B, 4] (motion)
        self.motion_mlp = nn.Sequential(
            nn.Linear(4, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Mish(),
            nn.Dropout(p=dropout_action),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Mish(),
            nn.Dropout(p=dropout_action),
        )

        # hand_action: [B, 16] (pose)
        self.pose_mlp = nn.Sequential(
            nn.Linear(16, action_hidden_dim),
            nn.BatchNorm1d(num_features=action_hidden_dim),
            nn.Mish(),
            nn.Dropout(p=dropout_action),
            nn.Linear(action_hidden_dim, action_embedding_dim),
            nn.BatchNorm1d(num_features=action_hidden_dim),
            nn.Mish(),
            nn.Dropout(p=dropout_action),
        )

        # Sound embedding
        self.sound_embedding = AudioCNN(
            embedding_dim=sound_embedding_dim,
            input_channels=1,  # Assuming mono audio input
        )

        # Final MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * 1024 + 2 * action_embedding_dim + sound_embedding_dim, final_hidden_dim),
            nn.BatchNorm1d(num_features=final_hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_final),
            nn.Linear(final_hidden_dim, output_dim),  # Output: [success_prob, gentleness_prob]
        )

    def _get_densenet(self, dropout: float = 0.5):
        model = densenet121(weights="IMAGENET1K_V1", drop_rate=dropout)
        model.classifier = nn.Identity()  # type: ignore
        return model

    def forward(self, input: dict[str, torch.Tensor],):
        vision_rgb = input["camera_rgb"]  # [B, 3, 224, 224]
        # vision_depth = input["camera_depth"]  # [B, 3, 224, 224]
        # touch_middle = input["touch_middle"]  # [B, 3, 224, 224]
        # touch_thumb = input["touch_thumb"]  # [B, 3, 224, 224]

        # *Note*: Zero tensors for actions as placeholders
        hand_action = torch.zeros(vision_rgb.shape[0], 16, device=vision_rgb.device)  # [B, 16]
        relpose_action = torch.zeros(vision_rgb.shape[0], 4, device=vision_rgb.device)  # [B, 4]
        

        # Extract features from DenseNet
        feat_rgb = self.vision_backbone_rgb(vision_rgb)  # [B, 1024]
        # feat_depth = self.vision_backbone_middle(vision_depth)
        feat_middle = self.vision_backbone_middle(input["touch_middle"])
        feat_thumb = self.vision_backbone_thumb(input["touch_thumb"])

        # Encode actions
        motion_feat = self.motion_mlp(relpose_action)  # [B, 64]
        pose_feat = self.pose_mlp(hand_action)  # [B, 64]

        sound = input["sound"].unsqueeze(1) # [B, 1, spectrogram_height, spectrogram_width]
        sound_feat = self.sound_embedding(sound)  # [B, sound_embedding_dim]

        # Concatenate all
        all_feats = torch.cat(
            [feat_rgb, feat_middle, feat_thumb, motion_feat, pose_feat, sound_feat],
            dim=1,
        )

        # Final MLP
        out = self.final_mlp(all_feats)  # [B, 2]
        return out


class StaticSoundAwareGraspSuccessModelModule(LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.model = hydra.utils.instantiate(self.cfg.model)  # type: ignore

        # Metrics
        metrics_raw = torchmetrics.MetricCollection(
            {"acc": BinaryAccuracy(), "specificity": BinarySpecificity()}
        )

        self.val_cm_joint = BinaryConfusionMatrix()

        self.output_labels = ["grasp"]

        for result_name in self.output_labels:
            for stage in ["train", "val"]:
                metrics = metrics_raw.clone(prefix=f"{stage}_{result_name}_")
                setattr(self, f"metrics_{stage}_{result_name}", metrics)

    def _compute_metrics(self, preds, labels, stage):
        preds_bin = (preds > 0.5).int()
        labels_bin = labels.int()

        for i, result_name in enumerate(self.output_labels):
            batch_values = getattr(self, f"metrics_{stage}_{result_name}")(
                preds_bin[:, i], labels_bin[:, i]
            )
            self.log_dict(batch_values, on_step=False, on_epoch=True, reduce_fx="mean")

    def _compute_joint_cm(self, preds, labels, stage):
        preds_bin = (preds > 0.5).int()
        labels_bin = labels.int()
        num_outputs = len(self.output_labels)

        if num_outputs == 1:
            pred_classes = preds_bin[:, 0]
            label_classes = labels_bin[:, 0]
        elif num_outputs == 2:
            pred_classes = preds_bin[:, 0] * 2 + preds_bin[:, 1]
            label_classes = labels_bin[:, 0] * 2 + labels_bin[:, 1]
        else:
            raise ValueError("Only supports 1 or 2 output labels for joint confusion matrix.")

        self.val_cm_joint.update(pred_classes, label_classes)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):

        inputs = batch
        labels = batch.pop("labels")
        preds = self(inputs)
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        self.log("train_loss", loss)

        self._compute_metrics(preds, labels, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        labels = batch.pop("labels")
        preds = self(inputs)
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        self.log("val_loss", loss, prog_bar=True)

        preds = F.sigmoid(preds)
        self._compute_metrics(preds, labels, "val")
        self._compute_joint_cm(preds, labels, "val")

    def on_validation_epoch_end(self):
        cm = self.val_cm_joint.compute()  # shape [4, 4]

        # Convert to DataFrame with meaningful labels
        labels = ["0", "1"]  # Assuming binary classification for grasp success
        cm_df = pd.DataFrame(cm.cpu().numpy(), index=labels, columns=labels)

        # Log to MLflow
        if isinstance(self.logger, MLFlowLogger):
            client: MlflowClient = self.logger.experiment
            client.log_table(
                run_id=self.logger.run_id,  # type: ignore
                artifact_file=f"confusion_matrix_epoch_{self.current_epoch}.json",
                data=cm_df,
            )

            # === Create and Log Plot ===
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Joint Confusion Matrix (Success + Gentleness)")

            with tempfile.TemporaryDirectory() as tmp_dir:
                plot_path = f"{tmp_dir}/confusion_matrix_epoch_{self.current_epoch}.png"
                fig.tight_layout()
                fig.savefig(plot_path)
                plt.close(fig)

                client.log_artifact(
                    run_id=self.logger.run_id,  # type: ignore
                    local_path=plot_path,
                    artifact_path="plots",
                )

        # Reset for next epoch
        self.val_cm_joint.reset()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer,  # type: ignore
        )(params=self.parameters())
        scheduler = hydra.utils.instantiate(
            self.cfg.lr_scheduler,  # type: ignore
        )(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
