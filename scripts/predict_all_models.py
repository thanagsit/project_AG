"""
scripts/predict_all_models.py
ระบบทำนายน้ำหนักหมูแบบ Full Pipeline (ประชัน 3 โมเดล!)
1. รับภาพต้นฉบับจากฟาร์ม
2. ใช้ YOLOv8_custom_pig ค้นหาและตีกรอบหมู
3. ส่งภาพเฉพาะส่วนที่ครอบได้ไปให้โมเดลทั้ง 3 ตัวแข่งกันทายน้ำหนัก:
   - ConvNeXt-Small
   - EfficientNet-B5
   - DINOv2-Small
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

# --- DINOv2 Architecture Definition ---
class PigModelDINOv2(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=False, force_reload=False
        )
        nf = 384
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.0),
            torch.nn.Linear(nf, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)

from models import build_model
from data import get_val_transforms
from utils import get_logger, load_checkpoint

logger = get_logger("predict_all")

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Pig Weight Prediction (All Models)")
    parser.add_argument("--image", type=str, required=True, help="Path to raw farm image")
    parser.add_argument("--yolo_weights", type=str, default="bestv8_last.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
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


from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_dinov2_transform(img_size=518):
    """Transform พื้นฐานสำหรับ DINOv2 (ใช้ค่า x-mean/std มาตรฐาน ImageNet)"""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

def load_model_ensemble(model_name: str, check_dir: str, cfg: OmegaConf, device: torch.device):
    """โหลดโมเดลตามชื่อและโฟลเดอร์ที่ระบุ (5 folds)"""
    cfg.model.backbone = model_name
    weight_paths = sorted(list(Path(check_dir).glob("best_fold*.pth")))
    if not weight_paths:
        logger.error(f"❌ ไม่พบ best_fold*.pth สำหรับ {model_name} ใน {check_dir}")
        return []

    models = []
    logger.setLevel("WARNING")
    for path in weight_paths:
        if model_name == "dinov2_small":
            model = PigModelDINOv2().to(device)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=True)
        else:
            model = build_model(cfg)
            load_checkpoint(model, str(path), device=device)
            model.to(device)
            
        model.eval()
        models.append(model)
    logger.setLevel("INFO")
    return models


def process_image(img_path, yolo_model, model_dict, cfg, args, device):
    print(f"\n============================================================")
    print(f"🔍 สแกนหาหมูด้วย YOLO ในภาพ: {img_path.name}")
    results = yolo_model(str(img_path), conf=args.yolo_conf, verbose=False)
    
    result = results[0]
    boxes = result.boxes
    
    if len(boxes) == 0:
        print("❌ ไม่พบหมูที่ผ่านเกณฑ์")
        return
        
    print(f"✅ พบหมูจำนวน {len(boxes)} ตัว!\n")
    
    orig_img = Image.open(img_path).convert("RGB")
    transform = get_val_transforms(cfg.data.img_size)
    cv_img = cv2.imread(str(img_path))
    
    for i, box in enumerate(boxes):
        conf = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        print(f"🐷 หมูตัวที่ {i+1} (ความมั่นใจ YOLO: {conf*100:.1f}%)")
        print(f"   ► พิกัด: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
        
        # Crop ตัดเฉพาะหมู
        cropped_img = orig_img.crop((x1, y1, x2, y2))
        
        # Transform ก้อนมาตรฐาน (224px) สำหรับ ConvNeXt และ B5
        image_tensor_standard = transform(image=np.array(cropped_img))["image"]
        
        # Transform ก้อนใหญ่พิเศษ (518px) สำหรับ DINOv2
        dino_transform = get_dinov2_transform(518)
        image_tensor_dino = dino_transform(image=np.array(cropped_img))["image"]
        
        # รันทำนายทีละรุ่น
        final_weights = {}
        for b_name, m_list in model_dict.items():
            if not m_list: continue
            
            # เลือกว่าจะป้อนรูปขนาดไหนให้โมเดล
            target_tensor = image_tensor_dino if "DINOv2" in b_name else image_tensor_standard
            
            all_preds = [predict_weight(m, target_tensor, device, args.tta) for m in m_list]
            final_weights[b_name] = sum(all_preds) / len(all_preds)
            print(f"   ⚖️  {b_name:15} ทายได้: {final_weights[b_name]:.2f} kg")
        
        # หาค่าเฉลี่ยจากทั้ง 3 โมเดลรวมกันเป็น Supreme Average
        if final_weights:
            supreme_avg = sum(final_weights.values()) / len(final_weights)
            print(f"   ⭐ ค่าเฉลี่ยรวม 3 โมเดล : {supreme_avg:.2f} kg")
            print("-" * 50)
            
            # วาดกรอบและป้ายแสดงค่าเฉลี่ย Supreme ลงบนภาพ
            cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
            text = f"Avg: {supreme_avg:.1f} kg"
            font = cv2.FONT_HERSHEY_DUPLEX; font_scale = 2.5; thickness = 5
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            tx, ty = int(x1), int(y1) - 15
            if ty < th: ty = int(y1) + th + 15
                
            cv2.rectangle(cv_img, (tx, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
            cv2.putText(cv_img, text, (tx + 5, ty), font, font_scale, (0, 255, 0), thickness)

    os.makedirs("outputs/predictions", exist_ok=True)
    out_path = f"outputs/predictions/compare_{img_path.name}"
    cv2.imwrite(out_path, cv_img)
    print(f"\n📸 บันทึกภาพแบบผสมผสานเสร็จสิ้นที่: {out_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)

    print("🚀 เริ่มระบบ Full Pipeline (ประชัน 3 โมเดล)...")
    
    # 1. YOLO
    yolo_model_path = Path(args.yolo_weights)
    if not yolo_model_path.exists():
        logger.error("❌ ไม่พบโมเดล YOLO")
        return
    yolo_model = YOLO(yolo_model_path)
    
    # 2. โหลดโมเดลทั้งหมด (ถ้าหาไม่เจอโฟลเดอร์ไหนจะข้ามไป)
    print("\n⏳ กำลังโหลดน้ำหนัก AI แต่ละสำนัก (ConvNeXt, B5, DINOv2)...")
    model_dict = {
        "ConvNeXt-S": load_model_ensemble("convnext_small", "outputs/checkpoints/convnext_small_hpc", cfg, device),
        "EfficientNet-B5": load_model_ensemble("efficientnet_b5", "outputs/checkpoints/b5", cfg, device),
        "DINOv2-S": load_model_ensemble("dinov2_small", "outputs/checkpoints/dinov2_small", cfg, device)
    }
    print("✅ โหลดสมองประมวลผลน้ำหนัก ครบถ้วน!\n")
    
    target_img = Path(args.image)
    if target_img.is_file():
        process_image(target_img, yolo_model, model_dict, cfg, args, device)
    elif target_img.is_dir():
        for img_file in sorted(target_img.glob("*.jpg")):
            process_image(img_file, yolo_model, model_dict, cfg, args, device)

    print("\n🎉 ปิดการประชัน เสร็จสิ้นภารกิจ!")

if __name__ == "__main__":
    main()
