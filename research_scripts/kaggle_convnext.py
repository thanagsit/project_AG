# ============================================================
# 🐷 Pig Weight — Train ConvNeXt-Small (Kaggle T4)
# ============================================================

# %%
# ─── CELL 1: Install Dependencies ───────────────────────────
import subprocess
subprocess.run(["pip", "install", "timm", "albumentations", "gdown", "rarfile", "-q"], check=True)
print("✅ Dependencies installed")


# %%
# ─── CELL 2: Load Dataset (Kaggle Dataset Input) ────────────
import os

KAGGLE_INPUT_DIR = "/kaggle/input/datasets/ppp123zz/pig-rgb-weight"

if not os.path.exists(KAGGLE_INPUT_DIR):
    raise FileNotFoundError(f"❌ ไม่พบ dataset ที่ {KAGGLE_INPUT_DIR}")

# หา root ที่ถูกต้อง
_candidates = [
    os.path.join(KAGGLE_INPUT_DIR, "RGB_9579"),
    os.path.join(KAGGLE_INPUT_DIR, "PigRGB_Weight", "RGB_9579"),
    os.path.join(KAGGLE_INPUT_DIR, "PigRGB_Weight"),
    KAGGLE_INPUT_DIR,
]
DATA_ROOT = next((p for p in _candidates if os.path.exists(p)), KAGGLE_INPUT_DIR)
print(f"✅ Dataset: {DATA_ROOT}")




# %%
# ─── CELL 3: Config ─────────────────────────────────────────
BACKBONE    = "convnext_small"
CKPT_SUBDIR = "convnext_small"

class Cfg:
    class data:
        root        = DATA_ROOT
        img_size    = 384      # เพิ่มความละเอียดเพื่อความแม่นยำ
        num_folds   = 5
        num_workers = 8        # H100 มักมี CPU core เยอะ
    class training:
        epochs      = 50
        batch_size  = 128      # H100 80GB รับได้สบาย (ปรับตามจริง)
        lr          = 1e-4
        weight_decay= 1e-5
        patience    = 15       # ให้เวลากับโมเดลที่ใหญ่ขึ้น
        use_amp     = True     # เปิดใช้ BF16/FP16 สำหรับ H100
    class output:
        dir         = "/kaggle/working/outputs"

cfg = Cfg()
print(f"✅ Config ready — backbone: {BACKBONE}")


# %%
# ─── CELL 4: Core Code ──────────────────────────────────────
import math, time, torch, torch.nn as nn, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

class PigDataset(Dataset):
    def __init__(self, fold_dirs, transform=None):
        self.transform = transform
        self.samples = []
        for fold_dir in fold_dirs:
            for pig_folder in sorted(Path(fold_dir).iterdir()):
                if not pig_folder.is_dir(): continue
                try: weight = float(pig_folder.name.split("_")[0])
                except ValueError: continue
                for img_file in sorted(pig_folder.glob("*.png")):
                    self.samples.append((str(img_file), weight))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, weight = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform: image = self.transform(image=image)["image"]
        return image, torch.tensor(weight, dtype=torch.float32)

# --- Aggressive Augmentations ในตัว ---
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_tf = A.Compose([
    A.Resize(cfg.data.img_size, cfg.data.img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),
        A.HueSaturationValue(p=1.0),
        A.ColorJitter(p=1.0),
    ], p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=1.0),
        A.GaussNoise(p=1.0),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
val_tf = A.Compose([
    A.Resize(cfg.data.img_size, cfg.data.img_size),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

def build_loaders(val_fold):
    all_folds = [os.path.join(cfg.data.root, f"fold{i}") for i in range(1, cfg.data.num_folds+1)]
    bs = cfg.training.batch_size
    tl = DataLoader(PigDataset([f for i,f in enumerate(all_folds,1) if i!=val_fold], transform=train_tf),
                    batch_size=bs, shuffle=True,  num_workers=cfg.data.num_workers, pin_memory=True)
    vl = DataLoader(PigDataset([all_folds[val_fold-1]], transform=val_tf),
                    batch_size=bs, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)
    return tl, vl

class PigModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model(BACKBONE, pretrained=True, num_classes=0, global_pool="avg")
        nf = self.encoder.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(nf,256), nn.ReLU(),
                                  nn.Dropout(dropout/2), nn.Linear(256,1))
    def forward(self, x): return self.head(self.encoder(x)).squeeze(1)

def compute_metrics(preds, targets):
    mae  = (preds - targets).abs().mean().item()
    rmse = math.sqrt(((preds - targets)**2).mean().item())
    ss_res = ((targets - preds)**2).sum().item()
    ss_tot = ((targets - targets.mean())**2).sum().item()
    return {"mae": mae, "rmse": rmse, "r2": 1.0 - ss_res/ss_tot if ss_tot>0 else 0.0}

print("✅ Core classes ready")


# %%
# ─── CELL 5: Train ──────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
torch.backends.cudnn.benchmark = True
print(f"🚀 Device: {device}  |  GPUs: {num_gpus}")

USE_AMP = cfg.training.use_amp and torch.cuda.is_available()
# H100 แนะนำใช้ bfloat16
scaler  = torch.amp.GradScaler("cuda", enabled=USE_AMP)
print(f"   AMP: {'ON (BF16/FP16)' if USE_AMP else 'OFF'}")

out_dir = Path(cfg.output.dir) / "checkpoints" / CKPT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

bs = cfg.training.batch_size * num_gpus if num_gpus > 1 else cfg.training.batch_size
cfg.training.batch_size = bs
print(f"   batch_size: {bs}")

criterion = nn.HuberLoss(delta=5.0)
fold_maes = []

for fold in range(5, cfg.data.num_folds + 1):  # รันแค่ Fold 5
    print(f"\n{'='*50}\n  [Fold {fold}/{cfg.data.num_folds}]\n{'='*50}")
    tl, vl = build_loaders(fold)
    model  = PigModel().to(device)
    if num_gpus > 1: model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    T_max=cfg.training.epochs, eta_min=1e-6)

    best_mae, patience_cnt = float("inf"), 0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        total = 0.0
        for imgs, targets in tqdm(tl, desc=f"Ep{epoch:03d}", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(targets)

        model.eval()
        preds_list, tgt_list = [], []
        with torch.no_grad():
            for imgs, targets in vl:
                preds_list.append(model(imgs.to(device)).cpu())
                tgt_list.append(targets)
        metrics = compute_metrics(torch.cat(preds_list), torch.cat(tgt_list))
        scheduler.step()

        print(f"  Ep {epoch:03d} | loss={total/len(tl.dataset):.4f} | MAE={metrics['mae']:.2f}kg | R²={metrics['r2']:.4f}")

        if metrics["mae"] < best_mae:
            best_mae, patience_cnt = metrics["mae"], 0
            raw = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"epoch": epoch, "model_state_dict": raw.state_dict(), "metrics": metrics},
                       out_dir / f"best_fold{fold}.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.training.patience:
                print(f"  ⏹ Early stop at epoch {epoch}")
                break

    fold_maes.append(best_mae)
    print(f"  ✅ Fold {fold} best MAE: {best_mae:.2f} kg")

print(f"\n🏁 {BACKBONE} — Avg MAE: {sum(fold_maes)/len(fold_maes):.2f} kg")


# %%
# ─── CELL 6: Pack & Download ────────────────────────────────
import shutil
shutil.make_archive("/kaggle/working/convnext_checkpoints", "zip", str(out_dir.parent.parent))
print("✅ Download: convnext_checkpoints.zip")
