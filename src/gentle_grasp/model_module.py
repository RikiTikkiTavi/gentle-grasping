from collections import defaultdict
from email.policy import default
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchvision.models import densenet121
from pytorch_lightning import LightningModule


class ActionConditionalModel(LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Vision backbones for 3 images
        self.vision_backbone_rgb = self._get_densenet()
        self.vision_backbone_middle = self._get_densenet()
        self.vision_backbone_thumb = self._get_densenet()

        # MLPs for action inputs
        # relpose_action: [B, 4] (motion)
        self.motion_mlp = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 64))
        # hand_action: [B, 16] (pose)
        self.pose_mlp = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 64))

        # Final MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(4 * 1024 + 2 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),  # Output: [success_prob, gentleness_prob]
            nn.Sigmoid(),
        )

        # Metrics
        metrics_raw = torchmetrics.MetricCollection({
            "acc": BinaryAccuracy(),
            "prec": BinaryPrecision(),
            "rec": BinaryRecall(),
        })

        for result_name in ["grasp", "gentle"]:
            for stage in ["train", "val"]:
                metrics = metrics_raw.clone(prefix=f"{stage}_{result_name}_")
                setattr(self, f"metrics_{stage}_{result_name}", metrics)

    def _get_densenet(self):
        model = densenet121(pretrained=True)
        model.classifier = nn.Identity()  # type: ignore
        return model

    def forward(
        self,
        vision_imgs: tuple[torch.Tensor, torch.Tensor],
        touch_imgs: tuple[torch.Tensor, torch.Tensor],
        actions: tuple[torch.Tensor, torch.Tensor],
    ):
        vision_rgb, vision_depth = vision_imgs  # [B, 3, 224, 224]
        touch_middle, touch_thumb = touch_imgs  # [B, 3, 224, 224]
        hand_action, relpose_action = actions  # [B, 16], [B, 4]

        # Flatten batch size 1 if needed (safe for both training and eval)
        B = vision_rgb.size(0)

        # Extract features from DenseNet
        feat_rgb = self.vision_backbone_rgb(vision_rgb)  # [B, 1024]
        feat_depth = self.vision_backbone_middle(vision_depth)
        feat_middle = self.vision_backbone_middle(touch_middle)
        feat_thumb = self.vision_backbone_thumb(touch_thumb)

        # Encode actions
        motion_feat = self.motion_mlp(relpose_action)  # [B, 64]
        pose_feat = self.pose_mlp(hand_action)  # [B, 64]

        # Concatenate all
        all_feats = torch.cat(
            [feat_rgb, feat_depth, feat_middle, feat_thumb, motion_feat, pose_feat],
            dim=1,
        )

        # Final MLP
        out = self.final_mlp(all_feats)  # [B, 2]
        return out

    def on_train_start(self):
        torch.set_float32_matmul_precision("high")

    def _compute_metrics(self, preds, labels, stage):
        preds_bin = (preds > 0.5).int()
        labels_bin = labels.int()

        for i, result_name in enumerate(["grasp", "gentle"]):
            batch_values = getattr(self, f"metrics_{stage}_{result_name}")(preds_bin[:, i], labels_bin[:, i])
            self.log_dict(batch_values, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        
        vision_imgs, touch_imgs, actions, labels = batch
        preds = self(vision_imgs, touch_imgs, actions)
        loss = F.binary_cross_entropy(preds, labels)
        self.log("train_loss", loss)

        self._compute_metrics(preds, labels, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        vision_imgs, touch_imgs, actions, labels = batch
        preds = self(vision_imgs, touch_imgs, actions)
        loss = F.binary_cross_entropy(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        
        self._compute_metrics(preds, labels, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
