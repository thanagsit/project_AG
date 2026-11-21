"""
utils/checkpoint.py
Save / Load model checkpoints
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """บันทึก model state + optimizer state + metrics"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    โหลด checkpoint กลับเข้า model

    Args:
        model: model instance (ต้องสร้างก่อน)
        path: path ของไฟล์ .pth
        optimizer: ถ้าต้องการโหลด optimizer state ด้วย (สำหรับ resume training)
        device: device เป้าหมาย

    Returns:
        dict ที่มี epoch และ metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    # รองรับทั้ง 2 key เพื่อ compatibility
    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
    model.load_state_dict(checkpoint[state_key])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
