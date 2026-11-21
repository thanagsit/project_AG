# ============================================================
# 🐷 Pig Weight Prediction — Kaggle Notebook (T4 GPU)
# Dataset from Google Drive
# ============================================================
# วิธีใช้: Copy ทั้งไฟล์นี้ไปวางใน Kaggle Notebook
# แล้วกด Run All (เปิด GPU T4 ที่ Settings > Accelerator ก่อน)
# ============================================================


# %%
# ─── CELL 1: Install Dependencies ───────────────────────────
import subprocess
subprocess.run(["pip", "install", "timm", "albumentations", "omegaconf", "gdown", "-q"], check=True)
print("✅ Dependencies installed")


# %%
# ─── CELL 2: Load Dataset ────────────────────────────────────
# รองรับ 2 วิธี:
#   A) Kaggle Dataset  → อัปโหลด ZIP ขึ้น kaggle.com/datasets แล้วแก้ชื่อด้านล่าง
#   B) Drive ZIP       → แชร์ไฟล์ .zip บน Drive แล้ววาง File ID ด้านล่าง

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [วิธี A] ถ้า upload ขึ้น Kaggle Dataset → แก้ชื่อ dataset ตรงนี้
KAGGLE_DATASET_NAME = "pig-rgb-weight"   # ← เปลี่ยนให้ตรงกับชื่อ dataset บน Kaggle
KAGGLE_INPUT_DIR    = f"/kaggle/input/{KAGGLE_DATASET_NAME}"

# [วิธี B] ถ้าใช้ไฟล์ RAR/ZIP จาก Google Drive → แก้ FILE_ID ตรงนี้
DRIVE_FILE_ID = "1NXddAksSKpvGHyhiXXKmel8ubMg20Krp"   # ← File ID ของไฟล์บน Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET_DIR = "/kaggle/working/PigRGB_Weight"

if os.path.exists(KAGGLE_INPUT_DIR):
    # ─── วิธี A: Kaggle Dataset ─────────────────────────────
    print(f"✅ [วิธี A] พบ Kaggle Dataset ที่: {KAGGLE_INPUT_DIR}")
    if os.path.exists(f"{KAGGLE_INPUT_DIR}/RGB_9579"):
        DATASET_DIR = KAGGLE_INPUT_DIR
    elif os.path.exists(f"{KAGGLE_INPUT_DIR}/PigRGB_Weight"):
        DATASET_DIR = f"{KAGGLE_INPUT_DIR}/PigRGB_Weight"
    else:
        DATASET_DIR = KAGGLE_INPUT_DIR
    print(f"   Dataset root → {DATASET_DIR}")

elif not os.path.exists(DATASET_DIR):
    # ─── วิธี B: Drive download + extract RAR ────────────────
    import subprocess, gdown

    # ติดตั้ง rarfile library และ unrar binary
    subprocess.run(["pip", "install", "rarfile", "-q"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-q", "unrar"], check=True)

    # download ไฟล์จาก Drive
    archive_path = "/kaggle/working/dataset.rar"
    print("⏳ [วิธี B] Downloading from Google Drive...")
    gdown.download(id=DRIVE_FILE_ID, output=archive_path, quiet=False, fuzzy=True)

    # extract RAR
    print("📦 Extracting RAR...")
    import rarfile
    with rarfile.RarFile(archive_path) as rf:
        rf.extractall("/kaggle/working/")
    os.remove(archive_path)
    print(f"✅ Extracted → /kaggle/working/")

else:
    print(f"✅ Dataset already exists at {DATASET_DIR}")


# %%
# ─── CELL 3: Config ─────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class DataConfig:
    root: str = f"{DATASET_DIR}/RGB_9579"
    img_size: int = 224
    num_folds: int = 5
    num_workers: int = 2      # Kaggle แนะนำ 2

@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    loss: str = "huber"

@dataclass
class OutputConfig:
    dir: str = "/kaggle/working/outputs"

class Cfg:
    data = DataConfig()
    model = ModelConfig()
    training = TrainingConfig()
    output = OutputConfig()

cfg = Cfg()
print("✅ Config loaded")
print(f"   Root   : {cfg.data.root}")
print(f"   Backbone: {cfg.model.backbone}")
print(f"   Epochs  : {cfg.training.epochs}")


# %%
# ─── CELL 4: Dataset & DataLoader ───────────────────────────
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class PigDataset(Dataset):
    """อ่าน label น้ำหนัก (kg) จากชื่อ folder: '114.5_138' → 114.5 kg"""

    def __init__(self, fold_dirs: List[str], transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, float]] = []
        for fold_dir in fold_dirs:
            fold_path = Path(fold_dir)
            for pig_folder in sorted(fold_path.iterdir()):
                if not pig_folder.is_dir():
                    continue
                try:
                    weight = float(pig_folder.name.split("_")[0])
                except ValueError:
                    continue
                for img_file in sorted(pig_folder.glob("*.png")):
                    self.samples.append((str(img_file), weight))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, weight = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(weight, dtype=torch.float32)


def build_fold_loaders(cfg, val_fold, train_tf, val_tf):
    all_folds = [os.path.join(cfg.data.root, f"fold{i}") for i in range(1, cfg.data.num_folds + 1)]
    val_folds  = [all_folds[val_fold - 1]]
    train_folds = [f for i, f in enumerate(all_folds, 1) if i != val_fold]

    train_ds = PigDataset(train_folds, transform=train_tf)
    val_ds   = PigDataset(val_folds,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size, shuffle=False,
                              num_workers=cfg.data.num_workers, pin_memory=True)
    return train_loader, val_loader

print("✅ Dataset class ready")


# %%
# ─── CELL 5: Augmentation ───────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20),
        ], p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

print("✅ Transforms ready")


# %%
# ─── CELL 6: Model ──────────────────────────────────────────
import torch.nn as nn
import timm

class PigWeightModel(nn.Module):
    def __init__(self, backbone="efficientnet_b3", pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        num_features = self.encoder.num_features  # 1536 for b3
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)

print("✅ Model class ready")


# %%
# ─── CELL 7: Loss & Metrics ─────────────────────────────────
import math

def build_loss(name="huber"):
    return {"huber": nn.HuberLoss(delta=5.0), "mae": nn.L1Loss(), "mse": nn.MSELoss()}[name]

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    mae  = (preds - targets).abs().mean().item()
    rmse = math.sqrt(((preds - targets) ** 2).mean().item())
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}

print("✅ Loss & Metrics ready")


# %%
# ─── CELL 8: Training Loop ──────────────────────────────────
from tqdm.notebook import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def val_one_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device)
        all_preds.append(model(images).cpu())
        all_targets.append(targets)
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))

print("✅ Training loop ready")


# %%
# ─── CELL 9: 5-Fold CV Training ─────────────────────────────
import pandas as pd

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = build_loss(cfg.training.loss)
train_tf  = get_train_transforms(cfg.data.img_size)
val_tf    = get_val_transforms(cfg.data.img_size)

os.makedirs(f"{cfg.output.dir}/checkpoints", exist_ok=True)

fold_results = []

for fold in range(1, cfg.data.num_folds + 1):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold} / {cfg.data.num_folds}  |  device: {device}")
    print(f"{'='*55}")

    train_loader, val_loader = build_fold_loaders(cfg, fold, train_tf, val_tf)
    print(f"  Train: {len(train_loader.dataset)} imgs | Val: {len(val_loader.dataset)} imgs")

    model     = PigWeightModel(cfg.model.backbone, cfg.model.pretrained, cfg.model.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    best_mae      = float("inf")
    patience_cnt  = 0
    ckpt_path     = f"{cfg.output.dir}/checkpoints/best_fold{fold}.pth"

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics    = val_one_epoch(model, val_loader, device)
        scheduler.step()

        mae, rmse, r2 = metrics["mae"], metrics["rmse"], metrics["r2"]
        print(f"  Epoch {epoch:03d}/{cfg.training.epochs} | loss={train_loss:.4f} | MAE={mae:.2f}kg | RMSE={rmse:.2f} | R²={r2:.4f}")

        if mae < best_mae:
            best_mae     = mae
            patience_cnt = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, ckpt_path)
            print(f"  ✅ Best saved → {ckpt_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.training.patience:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    fold_results.append({"fold": fold, "best_mae": best_mae})
    print(f"  ✅ Fold {fold} Best MAE: {best_mae:.2f} kg")


# %%
# ─── CELL 10: Summary ───────────────────────────────────────
df = pd.DataFrame(fold_results)
print("\n" + "="*55)
print("  📊 5-FOLD CROSS VALIDATION RESULTS")
print("="*55)
print(df.to_string(index=False))
print(f"\n  Avg MAE : {df['best_mae'].mean():.2f} ± {df['best_mae'].std():.2f} kg")
print(f"  Checkpoints saved in: {cfg.output.dir}/checkpoints/")
df.to_csv(f"{cfg.output.dir}/metrics.csv", index=False)
print("="*55)
