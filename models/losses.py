"""
models/losses.py
Loss functions สำหรับ regression task
"""

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1) — robust ต่อ outlier มากกว่า MSE
    delta=5 เพราะ weight range อยู่ที่ ~33–192 kg
    """

    def __init__(self, delta: float = 5.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


class MAELoss(nn.Module):
    """Mean Absolute Error — ตีความง่าย (ค่าผิดพลาดเฉลี่ยเป็น kg)"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


class MSELoss(nn.Module):
    """Mean Squared Error"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


def build_loss(loss_name: str) -> nn.Module:
    """Factory: สร้าง loss จาก string name"""
    loss_map = {
        "huber": HuberLoss(),
        "mae": MAELoss(),
        "mse": MSELoss(),
    }
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: '{loss_name}'. Options: {list(loss_map.keys())}")
    return loss_map[loss_name]
