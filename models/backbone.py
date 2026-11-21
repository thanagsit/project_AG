"""
models/backbone.py
PigWeightModel — EfficientNet-B3 backbone + regression head สำหรับทำนายน้ำหนักหมู (kg)
"""

import torch
import torch.nn as nn
import timm


class PigWeightModel(nn.Module):
    """
    EfficientNet-B3 backbone + custom regression head

    Architecture:
        EfficientNet-B3 (pretrained)
            → GlobalAvgPool (built-in)
            → Dropout(p)
            → Linear(1536 → 256) → ReLU → Dropout(p/2)
            → Linear(256 → 1)
            → output: น้ำหนัก (kg)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # โหลด backbone จาก timm (ไม่รวม classifier เดิม)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # ตัด classification head ออก
            global_pool="avg",      # GlobalAvgPool
        )
        num_features = self.encoder.num_features  # 1536 สำหรับ efficientnet_b3

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor [B, 3, H, W]
        Returns:
            weight prediction [B] (kg)
        """
        features = self.encoder(x)           # [B, 1536]
        out = self.head(features)            # [B, 1]
        return out.squeeze(1)                # [B]

    def freeze_backbone(self) -> None:
        """Freeze backbone สำหรับ warm-up training"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone สำหรับ full fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True


def build_model(cfg) -> PigWeightModel:
    """Factory function สร้าง model จาก OmegaConf config"""
    return PigWeightModel(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    )
