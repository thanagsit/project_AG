"""
training/trainer.py
Trainer class — จัดการ train loop, val loop, early stopping, scheduler
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import MetricTracker
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint

logger = get_logger(__name__)


class Trainer:
    """
    Encapsulates training và validation logic.

    Usage:
        trainer = Trainer(model, optimizer, criterion, scheduler, cfg, fold_num)
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        cfg,
        fold_num: int,
        scheduler=None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cfg = cfg
        self.fold_num = fold_num
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"[Fold {self.fold_num}] Train", leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        tracker = MetricTracker()

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            preds = self.model(images)
            tracker.update(preds, targets)

        return tracker.compute()

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        epochs = self.cfg.training.epochs
        patience = self.cfg.training.patience
        output_dir = Path(self.cfg.output.dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Fold {self.fold_num}] Starting training on {self.device} — {epochs} epochs")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.val_epoch(val_loader)

            val_mae = val_metrics["mae"]
            val_rmse = val_metrics["rmse"]
            val_r2 = val_metrics["r2"]

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_mae)
                else:
                    self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Fold {self.fold_num}] Epoch {epoch:03d}/{epochs} | "
                f"Loss: {train_loss:.4f} | MAE: {val_mae:.2f} kg | "
                f"RMSE: {val_rmse:.2f} | R²: {val_r2:.4f} | LR: {lr:.2e}"
            )

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_rmse"].append(val_rmse)
            self.history["val_r2"].append(val_r2)

            # Save best model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.patience_counter = 0
                ckpt_path = output_dir / f"best_fold{self.fold_num}.pth"
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, str(ckpt_path))
                logger.info(f"  ✅ Best model saved → {ckpt_path} (MAE={val_mae:.2f} kg)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"  ⏹ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        return self.history
