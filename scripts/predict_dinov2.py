"""
scripts/predict_dinov2.py
Inference เฉพาะสำหรับโมเดล DINOv2-Small ที่เทรนบน HPC
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_logger

logger = get_logger("predict_dinov2")


# ─── โครงสร้างโมเดล DINOv2 (Copy จากไฟล์ Train) ───
class PigModelDINOv2(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        # ใช้ torch.hub — หลีกปัญหา timm key mismatch
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=False, force_reload=False
        )
        nf = 384  # DINOv2 ViT-Small fixed
        
        # ถอด Dropout ออกตอน inference เพื่อให้ผลนิ่ง
        self.head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(nf, 256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # DINOv2 forward() คืน CLS token (B, 384)
        return self.head(self.encoder(x)).squeeze(1)


def get_dinov2_transform(img_size=518):
    """Transform พื้นฐานสำหรับ DINOv2 (ใช้ค่า x-mean/std มาตรฐาน ImageNet)"""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Pig weight prediction — DINOv2 Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to pig image (.png/.jpg)")
    parser.add_argument("--weights_dir", type=str, default="outputs/checkpoints/dinov2_small", help="Path to directory containing best_fold*.pth files")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--img_size", type=int, default=518, help="DINOv2 native patch14 resolution")
    return parser.parse_args()


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device, use_tta: bool) -> float:
    """ทำนายน้ำหนัก 1 โมเดล (รองรับ TTA)"""
    model.eval()
    with torch.no_grad():
        if not use_tta:
            batch = image_tensor.unsqueeze(0).to(device)
            pred = model(batch)
            return pred.item()
        
        # --- TTA (Test-Time Augmentation) ---
        # 1. Original
        # 2. Horizontal Flip
        batch = image_tensor.unsqueeze(0).to(device)
        batch_hf = torch.flip(batch, dims=[3])
        
        tta_batch = torch.cat([batch, batch_hf], dim=0)
        preds = model(tta_batch)
        return preds.mean().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. โหลดรูปภาพ
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error(f"❌ Image not found: {img_path}")
        return
        
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    transform = get_dinov2_transform(args.img_size)
    image_tensor = transform(image=image_np)["image"]

    # 2. ค้นหาไฟล์ Checkpoint (best_fold.pth)
    dir_path = Path(args.weights_dir)
    if not dir_path.exists():
        logger.error(f"❌ Directory not found: {dir_path}")
        return
        
    weight_paths = sorted(list(dir_path.glob("best_fold*.pth")))
    if not weight_paths:
        logger.error(f"❌ No best_fold*.pth files found in {args.weights_dir}")
        logger.error("👉 Please copy your trained DINOv2 checkpoints into this folder first.")
        return

    print(f"กำลังโหลดโมเดล DINOv2 จำนวน {len(weight_paths)} folds...")
    all_preds = []
    
    # 3. รันทีละ Fold
    for path in weight_paths:
        model = PigModelDINOv2().to(device)
        
        # โหลด weights (รองรับไฟล์ที่เซฟมาจากทั้ง HPC GPU และ CPU)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # ตรวจสอบว่ามี dict ซ้อนไหม
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # ลบ prefix '_orig_mod.' ออก (ถ้าตอนเทรนใน HPC มีการใช้ torch.compile)
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("_orig_mod.", "")
            clean_state_dict[new_k] = v
            
        model.load_state_dict(clean_state_dict, strict=True)
        
        # ทำนาย
        pred_val = predict(model, image_tensor, device, args.tta)
        all_preds.append(pred_val)
        
        del model
        torch.cuda.empty_cache()

    final_weight = sum(all_preds) / len(all_preds)

    # 4. แสดงผลลัพธ์
    print(f"\n{'='*55}")
    print(f"  🦖 DINOv2-Small 142M (Meta AI)")
    print(f"  🐷 ภาพ: {img_path.name}")
    print(f"  ⚙️  Ensemble: {len(weight_paths)} Folds")
    print(f"  🚀 TTA: {'✅ ใช่ (พลิกซ้ายขวา)' if args.tta else '❌ ไม่'}")
    print(f"  ⚖️  น้ำหนักที่ทำนายได้: {final_weight:.2f} kg")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
