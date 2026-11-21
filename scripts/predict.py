"""
scripts/predict.py
Inference บนรูปภาพเดียว → แสดงน้ำหนักที่ทำนาย (kg) พร้อมรองรับ 5-Fold Ensemble และ TTA

Usage:
    python scripts/predict.py --image path/to/pig.png --weights outputs/checkpoints/convnext_small/best_fold1.pth
    python scripts/predict.py --image path/to/pig.png --weights_dir outputs/checkpoints/convnext_small --tta
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model
from data import get_val_transforms
from utils import get_logger, load_checkpoint

logger = get_logger("predict")


def parse_args():
    parser = argparse.ArgumentParser(description="Pig weight prediction — single image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to pig image (.png/.jpg)")
    parser.add_argument("--backbone", type=str, default="convnext_small", help="Model backbone to use")
    parser.add_argument("--weights", type=str, default=None, help="Path to a single .pth checkpoint file")
    parser.add_argument("--weights_dir", type=str, default=None, help="Path to directory containing multiple fold weights for ensemble")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device, use_tta: bool) -> float:
    """ทำนายน้ำหนักจาก 1 โมเดล (รองรับ TTA)"""
    model.eval()
    with torch.no_grad():
        if not use_tta:
            # รูปเดียวปกติ
            batch = image_tensor.unsqueeze(0).to(device)
            pred = model(batch)
            return pred.item()
        
        # --- TTA (Test-Time Augmentation) ---
        # 1. Original
        # 2. Horizontal Flip
        # 3. Vertical Flip
        # 4. Rotate 180 (H + V Flip)
        batch = image_tensor.unsqueeze(0).to(device)
        batch_hf = torch.flip(batch, dims=[3])
        batch_vf = torch.flip(batch, dims=[2])
        batch_rot = torch.flip(batch, dims=[2, 3])
        
        tta_batch = torch.cat([batch, batch_hf, batch_vf, batch_rot], dim=0) # [4, 3, H, W]
        preds = model(tta_batch)
        return preds.mean().item()


def main():
    args = parse_args()
    
    if args.weights is None and args.weights_dir is None:
        logger.error("Must provide either --weights or --weights_dir")
        return

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model.backbone = args.backbone

    # Load 1 image
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        return
        
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    transform = get_val_transforms(cfg.data.img_size)
    image_tensor = transform(image=image_np)["image"]

    # Gather weights to load
    weight_paths = []
    if args.weights_dir:
        dir_path = Path(args.weights_dir)
        weight_paths = sorted(list(dir_path.glob("best_fold*.pth")))
        if not weight_paths:
            logger.error(f"No best_fold*.pth files found in {args.weights_dir}")
            return
    else:
        weight_paths = [Path(args.weights)]

    # Suppress verbose loads if multiple models
    if len(weight_paths) > 1:
        logger.setLevel("WARNING")

    all_preds = []
    
    for path in weight_paths:
        model = build_model(cfg)
        load_checkpoint(model, str(path), device=device)
        model.to(device)
        
        pred_val = predict(model, image_tensor, device, args.tta)
        all_preds.append(pred_val)
        
        # Free memory if multiple models
        del model
        torch.cuda.empty_cache()

    final_weight = sum(all_preds) / len(all_preds)

    print(f"\n{'='*45}")
    print(f"  🐷 ภาพ: {img_path.name}")
    print(f"  ⚙️  โมเดลที่ใช้: {len(weight_paths)} Folds Ensemble")
    print(f"  🚀 เปิด TTA: {'✅ ใช่ (มอง 4 มุม)' if args.tta else '❌ ไม่'}")
    print(f"  ⚖️  น้ำหนักที่ทำนาย (สรุป): {final_weight:.2f} kg")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
