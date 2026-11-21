"""
scripts/evaluate.py
ประเมินโมเดลที่บันทึกไว้ — รองรับ 2 โหมด:

  โหมดปกติ (single backbone):
      python scripts/evaluate.py
      python scripts/evaluate.py --fold 1

  โหมด compare หลาย backbone:
      python scripts/evaluate.py --compare
      # จะอ่าน checkpoints จาก outputs/checkpoints/<backbone>/
      # และเปรียบเทียบทุกโมเดลพร้อมกัน รวม B3 baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import pandas as pd
from omegaconf import OmegaConf

from data import build_fold_loaders, get_train_transforms, get_val_transforms
from models import build_model
from training import Trainer
from utils import get_logger, load_checkpoint

logger = get_logger("evaluate")

# ── โมเดลที่รู้จัก — เพิ่มได้เรื่อยๆ ────────────────────────
KNOWN_MODELS = {
    "efficientnet_b3": {
        "name": "EfficientNet-B3",
        "ckpt_dir": "checkpoints/efficientnet_b3",
    },
    "efficientnet_b5": {
        "name": "EfficientNet-B5",
        "ckpt_dir": "checkpoints/efficientnet_b5",
    },
    "convnext_small": {
        "name": "ConvNeXt-Small",
        "ckpt_dir": "checkpoints/convnext_small",
    },
    "convnext_small_hpc": {
        "name": "ConvNeXt-Small (HPC)",
        "ckpt_dir": "checkpoints/convnext_small_hpc",
        "backbone": "convnext_small"  # 🐷 timm ต้องใช้ชื่อนี้
    },
    "vit_small_patch14_dinov2.lvd142m": {
        "name": "DINOv2-Small",
        "ckpt_dir": "checkpoints/dinov2_small",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config/config.yaml")
    p.add_argument("--fold",    type=int, default=None)
    p.add_argument("--compare", action="store_true",
                   help="เปรียบเทียบทุก backbone ที่มี checkpoint")
    return p.parse_args()


@torch.no_grad()
def evaluate_model(cfg, backbone: str, ckpt_dir: str, folds, device, val_tf, train_tf):
    """โหลด + evaluate โมเดลหนึ่ง backbone ทุก fold → return DataFrame"""
    out_root = Path(cfg.output.dir)
    results  = []

    for fold in folds:
        ckpt = out_root / ckpt_dir / f"best_fold{fold}.pth"
        if not ckpt.exists():
            logger.warning(f"  ไม่พบ {ckpt} — ข้าม")
            continue

        _, val_loader = build_fold_loaders(
            data_root=cfg.data.root,
            val_fold=fold,
            num_folds=cfg.data.num_folds,
            train_transform=train_tf,
            val_transform=val_tf,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )

        # build model ด้วย backbone ที่ระบุ, pretrained=False (load จาก ckpt)
        eval_cfg = OmegaConf.merge(cfg, OmegaConf.create({"model": {
            "backbone": backbone, "pretrained": False
        }}))
        model = build_model(eval_cfg)
        info  = load_checkpoint(model, str(ckpt), device=device)
        model.to(device)

        trainer = Trainer(model=model, optimizer=None, criterion=None,
                          cfg=cfg, fold_num=fold, device=device)
        metrics = trainer.val_epoch(val_loader)

        logger.info(
            f"  [Fold {fold}] epoch={info['epoch']:>3}  "
            f"MAE={metrics['mae']:.2f} kg  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.4f}"
        )
        results.append({"fold": fold, **metrics})

    return pd.DataFrame(results) if results else pd.DataFrame()


def run_compare(cfg, folds, device, val_tf, train_tf):
    """เปรียบเทียบทุก backbone ที่มี checkpoint อยู่"""
    summary = []

    for backbone, info in KNOWN_MODELS.items():
        ckpt_root = Path(cfg.output.dir) / info["ckpt_dir"]
        # ข้ามถ้าไม่มี checkpoint เลย
        if not any(ckpt_root.glob("best_fold*.pth")):
            logger.info(f"\n[{info['name']}] ไม่พบ checkpoint → ข้าม")
            continue

        logger.info(f"\n{'='*55}")
        logger.info(f"  {info['name']}  ({backbone})")
        logger.info(f"{'='*55}")

        real_backbone = info.get("backbone", backbone)
        df = evaluate_model(cfg, real_backbone, info["ckpt_dir"], folds, device, val_tf, train_tf)
        if df.empty:
            continue

        avg_mae  = df["mae"].mean()
        avg_rmse = df["rmse"].mean()
        avg_r2   = df["r2"].mean()
        logger.info(f"\n  Avg → MAE={avg_mae:.2f} kg  RMSE={avg_rmse:.2f}  R²={avg_r2:.4f}")

        summary.append({
            "model":    info["name"],
            "backbone": real_backbone,
            "avg_mae":  round(avg_mae, 3),
            "avg_rmse": round(avg_rmse, 3),
            "avg_r2":   round(avg_r2, 4),
            "std_mae":  round(df["mae"].std(), 3),
        })

    if not summary:
        logger.warning("ไม่มีโมเดลให้เปรียบเทียบ")
        return

    result_df = pd.DataFrame(summary).sort_values("avg_mae")
    winner    = result_df.iloc[0]

    print(f"\n{'='*55}")
    print("  🏆 MODEL COMPARISON")
    print(f"{'='*55}")
    print(result_df[["model","avg_mae","std_mae","avg_rmse","avg_r2"]].to_string(index=False))
    print(f"\n  🥇 Winner: {winner['model']}  MAE={winner['avg_mae']:.2f} kg  R²={winner['avg_r2']:.4f}")

    # บันทึก CSV
    csv_path = Path(cfg.output.dir) / "logs" / "model_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")


def main():
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_tf   = get_val_transforms(cfg.data.img_size)
    train_tf = get_train_transforms(cfg.data.img_size)
    folds    = [args.fold] if args.fold else list(range(1, cfg.data.num_folds + 1))

    if args.compare:
        # ─── โหมด compare ─────────────────────────────────────
        run_compare(cfg, folds, device, val_tf, train_tf)
    else:
        # ─── โหมดปกติ: B3 (default backbone) ─────────────────
        backbone = cfg.model.backbone
        ckpt_dir = KNOWN_MODELS.get(backbone, {}).get("ckpt_dir", "checkpoints")

        df = evaluate_model(cfg, backbone, ckpt_dir, folds, device, val_tf, train_tf)
        if not df.empty:
            logger.info(
                f"\nAvg MAE: {df['mae'].mean():.2f} kg | "
                f"Avg RMSE: {df['rmse'].mean():.2f} | "
                f"Avg R²: {df['r2'].mean():.4f}"
            )


if __name__ == "__main__":
    main()
