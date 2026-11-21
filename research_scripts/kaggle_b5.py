# ============================================================
# 🐷 Pig Weight — Train EfficientNet-B5 (Kaggle T4)
# ============================================================

# %%
# ─── CELL 1: Install Dependencies ───────────────────────────
import subprocess
subprocess.run(["pip", "install", "timm", "albumentations", "gdown", "rarfile", "-q"], check=True)
print("✅ Dependencies installed")


# %%
# ─── CELL 2: Load Dataset ───────────────────────────────────
import os

KAGGLE_DATASET_NAME = "pig-rgb-weight"
KAGGLE_INPUT_DIR    = f"/kaggle/input/{KAGGLE_DATASET_NAME}"
DRIVE_FILE_ID       = "1NXddAksSKpvGHyhiXXKmel8ubMg20Krp"
DATASET_DIR         = "/kaggle/working/PigRGB_Weight"

if os.path.exists(KAGGLE_INPUT_DIR):
    print(f"✅ Kaggle Dataset: {KAGGLE_INPUT_DIR}")
    DATASET_DIR = (KAGGLE_INPUT_DIR + "/RGB_9579" if os.path.exists(f"{KAGGLE_INPUT_DIR}/RGB_9579")
                   else KAGGLE_INPUT_DIR + "/PigRGB_Weight" if os.path.exists(f"{KAGGLE_INPUT_DIR}/PigRGB_Weight")
                   else KAGGLE_INPUT_DIR)
elif not os.path.exists(DATASET_DIR):
    import gdown, rarfile
    subprocess.run(["apt-get", "install", "-y", "-q", "unrar"], check=True)
    archive_path = "/kaggle/working/dataset.rar"
    print("⏳ Downloading from Google Drive...")
    gdown.download(id=DRIVE_FILE_ID, output=archive_path, quiet=False, fuzzy=True)
    print("📦 Extracting...")
    with rarfile.RarFile(archive_path) as rf:
        rf.extractall("/kaggle/working/")
    os.remove(archive_path)
else:
    print(f"✅ Dataset exists: {DATASET_DIR}")

print(f"   Dataset root → {DATASET_DIR}")


# %%
# ─── CELL 3: Config ─────────────────────────────────────────
BACKBONE    = "efficientnet_b5"
CKPT_SUBDIR = "efficientnet_b5"   # outputs/checkpoints/efficientnet_b5/

class Cfg:
    class data:
        root        = f"{DATASET_DIR}/RGB_9579" if os.path.exists(f"{DATASET_DIR}/RGB_9579") else DATASET_DIR
        img_size    = 224
        num_folds   = 5
        num_workers = 2
    class training:
        epochs      = 50
        batch_size  = 32
        lr          = 1e-4
        weight_decay= 1e-5
        patience    = 10
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

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_tf = A.Compose([A.Resize(224,224), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
                      A.RandomRotate90(p=0.3), A.RandomBrightnessContrast(p=0.3),
                      A.Normalize(mean=MEAN,std=STD), ToTensorV2()])
val_tf   = A.Compose([A.Resize(224,224), A.Normalize(mean=MEAN,std=STD), ToTensorV2()])

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
torch.backends.cudnn.benchmark = False
print(f"🚀 Device: {device}  |  GPUs: {num_gpus}")

out_dir = Path(cfg.output.dir) / "checkpoints" / CKPT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

bs = cfg.training.batch_size * num_gpus if num_gpus > 1 else cfg.training.batch_size
cfg.training.batch_size = bs
print(f"   batch_size: {bs}")

criterion  = nn.HuberLoss(delta=5.0)
fold_maes  = []

for fold in range(1, cfg.data.num_folds + 1):
    print(f"\n{'='*50}\n  [Fold {fold}/{cfg.data.num_folds}]\n{'='*50}")
    tl, vl = build_loaders(fold)
    model  = PigModel().to(device)
    if num_gpus > 1: model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    best_mae, patience_cnt = float("inf"), 0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        total = 0.0
        for imgs, targets in tqdm(tl, desc=f"Ep{epoch:03d}", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward(); optimizer.step()
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
shutil.make_archive(f"/kaggle/working/b5_checkpoints", "zip", str(out_dir.parent.parent))
print("✅ Download: b5_checkpoints.zip")
