"""
training/metrics.py
Regression metrics: MAE, RMSE, R²
"""

import torch
import math


def mean_absolute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """MAE — ค่าผิดพลาดเฉลี่ยเป็น kg"""
    return (preds - targets).abs().mean().item()


def root_mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """RMSE"""
    mse = ((preds - targets) ** 2).mean().item()
    return math.sqrt(mse)


def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    R² (Coefficient of Determination)
    R²=1.0 คือทำนายสมบูรณ์แบบ, R²=0 คือทำนายแย่เท่า mean baseline
    """
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


class MetricTracker:
    """สะสม predictions และ targets ระหว่าง epoch แล้วคำนวณ metrics รวม"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds = []
        self._targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        return {
            "mae": mean_absolute_error(preds, targets),
            "rmse": root_mean_squared_error(preds, targets),
            "r2": r2_score(preds, targets),
        }
