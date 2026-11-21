"""
data/dataset.py
PigDataset — อ่านภาพ RGB และ label น้ำหนัก (kg) จากชื่อ folder

โครงสร้าง folder ที่รองรับ:
  RGB_9579/
    fold1/
      101.74_127/       ← <weight_kg>_<num_images>
        101.74kg_1.png
        ...
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image


class PigDataset(Dataset):
    """Dataset สำหรับโหลดภาพหมูและน้ำหนัก (kg) จากโครงสร้าง folder"""

    def __init__(
        self,
        fold_dirs: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            fold_dirs: list ของ path ไปยัง fold folder ที่ต้องการใช้
                       เช่น ["RGB_9579/fold1", "RGB_9579/fold2"]
            transform: albumentations transform pipeline
        """
        self.transform = transform
        self.samples: List[Tuple[str, float]] = []
        self._build_samples(fold_dirs)

    def _build_samples(self, fold_dirs: List[str]) -> None:
        """สแกนทุก fold และ build รายการ (image_path, weight_kg)"""
        for fold_dir in fold_dirs:
            fold_path = Path(fold_dir)
            if not fold_path.exists():
                raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

            for pig_folder in sorted(fold_path.iterdir()):
                if not pig_folder.is_dir():
                    continue

                weight_kg = self._parse_weight(pig_folder.name)
                if weight_kg is None:
                    continue

                for img_file in sorted(pig_folder.glob("*.png")):
                    self.samples.append((str(img_file), weight_kg))

    @staticmethod
    def _parse_weight(folder_name: str) -> Optional[float]:
        """
        Parse น้ำหนักจากชื่อ folder
        ตัวอย่าง: "101.74_127" → 101.74
        """
        try:
            return float(folder_name.split("_")[0])
        except (ValueError, IndexError):
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, weight = self.samples[idx]

        # โหลดภาพเป็น RGB เสมอ
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            import numpy as np
            image = self.transform(image=np.array(image))["image"]
        else:
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)

        label = torch.tensor(weight, dtype=torch.float32)
        return image, label

    def get_weight_stats(self) -> dict:
        """คำนวณ min/max/mean weight สำหรับ sanity check"""
        weights = [w for _, w in self.samples]
        return {
            "count": len(weights),
            "min": min(weights),
            "max": max(weights),
            "mean": sum(weights) / len(weights),
        }


def build_fold_loaders(
    data_root: str,
    val_fold: int,
    num_folds: int,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    สร้าง DataLoader สำหรับ 1 fold round ของ 5-Fold CV

    Args:
        data_root: path ไปยัง RGB_9579/
        val_fold: fold ที่ใช้เป็น validation (1-indexed)
        num_folds: จำนวน fold ทั้งหมด
        ...
    """
    all_folds = [os.path.join(data_root, f"fold{i}") for i in range(1, num_folds + 1)]
    val_folds = [all_folds[val_fold - 1]]
    train_folds = [f for i, f in enumerate(all_folds, 1) if i != val_fold]

    train_dataset = PigDataset(fold_dirs=train_folds, transform=train_transform)
    val_dataset = PigDataset(fold_dirs=val_folds, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
