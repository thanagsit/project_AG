"""
scripts/train.py
Entry point — รัน 5-Fold Cross Validation Training

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --epochs 1 --debug
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from omegaconf import OmegaConf

from data import build_fold_loaders, get_train_transforms, get_val_transforms
from models import build_model, build_loss
from training import Trainer
from utils import get_logger

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Pig Weight Prediction — 5-Fold CV Training")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
    parser.add_argument("--debug", action="store_true", help="Run 1 epoch for smoke test")
    return parser.parse_args()


def build_optimizer_and_scheduler(model, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=1e-6
        )
    elif cfg.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    return optimizer, scheduler


def main():
    args = parse_args()

    # โหลด config
    cfg = OmegaConf.load(args.config)

    # Override จาก args
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.debug:
        cfg.training.epochs = 1
        cfg.training.patience = 999
        logger.info("🐛 DEBUG mode: 1 epoch only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    logger.info(f"📁 Data root: {cfg.data.root}")

    train_tf = get_train_transforms(cfg.data.img_size)
    val_tf = get_val_transforms(cfg.data.img_size)
    criterion = build_loss(cfg.training.loss)

    fold_results = []
    start_time = time.time()

    for fold in range(1, cfg.data.num_folds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD {fold} / {cfg.data.num_folds}")
        logger.info(f"{'='*60}")

        train_loader, val_loader = build_fold_loaders(
            data_root=cfg.data.root,
            val_fold=fold,
            num_folds=cfg.data.num_folds,
            train_transform=train_tf,
            val_transform=val_tf,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )

        logger.info(f"  Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images")

        model = build_model(cfg)
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            cfg=cfg,
            fold_num=fold,
            device=device,
        )

        history = trainer.fit(train_loader, val_loader)

        best_mae = min(history["val_mae"])
        best_rmse = history["val_rmse"][history["val_mae"].index(best_mae)]
        best_r2 = history["val_r2"][history["val_mae"].index(best_mae)]

        fold_results.append({
            "fold": fold,
            "best_mae": best_mae,
            "best_rmse": best_rmse,
            "best_r2": best_r2,
        })

        logger.info(f"  ✅ Fold {fold} Best → MAE: {best_mae:.2f} kg | RMSE: {best_rmse:.2f} | R²: {best_r2:.4f}")

    # สรุปผลทุก fold
    df = pd.DataFrame(fold_results)
    log_dir = Path(cfg.output.dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_dir / "metrics.csv", index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("  📊 5-FOLD CROSS VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Avg MAE  : {df['best_mae'].mean():.2f} ± {df['best_mae'].std():.2f} kg")
    logger.info(f"  Avg RMSE : {df['best_rmse'].mean():.2f} ± {df['best_rmse'].std():.2f}")
    logger.info(f"  Avg R²   : {df['best_r2'].mean():.4f} ± {df['best_r2'].std():.4f}")
    logger.info(f"  Total time: {elapsed/60:.1f} min")
    logger.info(f"  Results saved → {log_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
