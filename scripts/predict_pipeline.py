"""
scripts/predict_pipeline.py
ระบบทำนายน้ำหนักหมูแบบ Full Pipeline (End-to-End) พร้อมวาดรูปลงบนภาพจริง
1. รับภาพต้นฉบับจากฟาร์ม
2. ใช้ YOLOv8_custom_pig ค้นหาและตีกรอบหมู
3. ส่งภาพเฉพาะส่วนที่ครอบได้ไปให้ ConvNeXt / DINOv2 ทายน้ำหนัก
4. วาดกรอบและตัวหนังสือ (น้ำหนัก) ทับลงบนรูปต้นฉบับ แล้วเซฟผลลัพธ์
"""

import sys
import os
import cv2
from pathlib import Path
import argparse
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model
from data import get_val_transforms
from utils import get_logger, load_checkpoint

logger = get_logger("predict_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Pig Weight Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to raw farm image (.jpg/.png)")
    parser.add_argument("--yolo_weights", type=str, default="bestv8_last.pt", help="Path to custom YOLOv8 weights")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--weight_backbone", type=str, default="convnext_small", help="Weight model backbone")
    parser.add_argument("--weight_model_dir", type=str, default="outputs/checkpoints/convnext_small_hpc", help="Directory containing best_fold*.pth for weight model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--tta", action="store_true", help="Enable TTA on weight prediction")
    return parser.parse_args()


def predict_weight(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device, use_tta: bool) -> float:
    model.eval()
    with torch.no_grad():
        if not use_tta:
            batch = image_tensor.unsqueeze(0).to(device)
            pred = model(batch)
            return pred.item()
        
        batch = image_tensor.unsqueeze(0).to(device)
        batch_hf = torch.flip(batch, dims=[3])
        batch_vf = torch.flip(batch, dims=[2])
        batch_rot = torch.flip(batch, dims=[2, 3])
        
        tta_batch = torch.cat([batch, batch_hf, batch_vf, batch_rot], dim=0)
        preds = model(tta_batch)
        return preds.mean().item()


def load_weight_ensemble(args, cfg, device):
    weight_paths = sorted(list(Path(args.weight_model_dir).glob("best_fold*.pth")))
    if not weight_paths:
        logger.error(f"❌ No best_fold*.pth found in {args.weight_model_dir}")
        sys.exit(1)

    print(f"กำลังประกอบร่างโมเดลน้ำหนัก (Ensemble {len(weight_paths)} Folds)...")
    models = []
    
    logger.setLevel("WARNING")
    for path in weight_paths:
        model = build_model(cfg)
        load_checkpoint(model, str(path), device=device)
        model.to(device)
        model.eval()
        models.append(model)
        
    logger.setLevel("INFO")
    return models


def process_image(img_path, yolo_model, weight_models, cfg, args, device):
    print(f"\n🔍 1. สแกนหาหมูด้วย YOLO ในภาพ: {img_path.name}")
    results = yolo_model(str(img_path), conf=args.yolo_conf, verbose=False)
    
    result = results[0]
    boxes = result.boxes
    
    if len(boxes) == 0:
        print("❌ ไม่พบหมูที่ผ่านเกณฑ์ (Confidence ต่ำไป หรือไม่มีหมูในภาพ)")
        return
        
    print(f"✅ พบหมูจำนวน {len(boxes)} ตัว!")
    
    orig_img = Image.open(img_path).convert("RGB")
    transform = get_val_transforms(cfg.data.img_size)
    
    # อ่านรูปภาพต้นฉบับด้วย OpenCV เพื่อไว้วาดกรอบ (ใช้ cv2.imread)
    cv_img = cv2.imread(str(img_path))
    
    for i, box in enumerate(boxes):
        conf = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        print(f"\n🐷 หมูตัวที่ {i+1} (YOLO Conf: {conf*100:.1f}%)")
        print(f"   ► พิกัดกล่อง: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
        
        # Crop ตัดเฉพาะหมู
        cropped_img = orig_img.crop((x1, y1, x2, y2))
        
        # รันทำนายน้ำหนักจากรูปที่ตัดมาแล้ว
        cropped_np = np.array(cropped_img)
        image_tensor = transform(image=cropped_np)["image"]
        
        all_preds = []
        for w_model in weight_models:
            pred_val = predict_weight(w_model, image_tensor, device, args.tta)
            all_preds.append(pred_val)
            
        final_weight = sum(all_preds) / len(all_preds)
        print(f"   ⚖️ น้ำหนักที่ทำนายได้: {final_weight:.2f} kg")
        
        # ----- วาดกรอบและข้อความลงไปบนรูปต้นฉบับ -----
        # 1. กล่องสี่เหลี่ยม (สีเขียว ความหนา 5px)
        cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        
        # 2. ป้ายข้อความน้ำหนัก (พร้อมพื้นหลังสีดำทึบเพื่อให้ตัวอักษรเด่นชัด)
        text = f"{final_weight:.1f} kg"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 3.0
        thickness = 5
        
        # หาขนาดกรอบตัวอักษรเพื่อวาดพื้นหลัง
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # ตำแหน่งมุมซ้ายบนของป้ายชื่อ
        text_x = int(x1)
        text_y = int(y1) - 15  # ขยับขึ้นไปเหนือกล่องนิดนึง
        if text_y < text_height:  # ถ้าชนขอบบน ให้ย้ายป้ายมาไว้ข้างในกล่อง
            text_y = int(y1) + text_height + 15
            
        cv2.rectangle(cv_img, (text_x, text_y - text_height - 10), (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(cv_img, text, (text_x + 5, text_y), font, font_scale, (0, 255, 0), thickness)
        # ---------------------------------------------

    # เซฟรูปภาพที่วาดเสร็จแล้ว
    os.makedirs("outputs/predictions", exist_ok=True)
    out_path = f"outputs/predictions/final_{img_path.name}"
    cv2.imwrite(out_path, cv_img)
    print(f"\n📸 บันทึกภาพพร้อมกรอบน้ำหนักเสร็จสิ้นที่: {out_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 เริ่มระบบ Full Pipeline Detection-to-Weight...")
    yolo_model_path = Path(args.yolo_weights)
    if not yolo_model_path.exists():
        logger.error(f"❌ ไม่พบโมเดล YOLO ที่ path: {yolo_model_path}")
        return
    yolo_model = YOLO(yolo_model_path)
    print("✅ โหลดโมเดล YOLO สำเร็จ")

    cfg = OmegaConf.load(args.config)
    cfg.model.backbone = args.weight_backbone
    weight_models = load_weight_ensemble(args, cfg, device)
    print("✅ โหลดโมเดลคาดเดาน้ำหนัก สำเร็จ")
    print("="*60)
    
    target_img = Path(args.image)
    if target_img.is_file():
        process_image(target_img, yolo_model, weight_models, cfg, args, device)
    elif target_img.is_dir():
        for img_file in sorted(target_img.glob("*.jpg")):
            process_image(img_file, yolo_model, weight_models, cfg, args, device)
            print("-"*40)

    print("\n🎉 เสร็จสิ้นภารกิจ!")

if __name__ == "__main__":
    main()
