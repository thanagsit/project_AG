import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import timm
import cv2
from ultralytics import YOLO
from glob import glob

# ─────────────────────────────────────────────────────────────
# 🛠️ Configuration
# ─────────────────────────────────────────────────────────────
IMAGE_PATH = r"..\test_images\farm_test_05.jpg"
OUTPUT_DIR = r"..\outputs\checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# 🐷 Model Architectures
# ─────────────────────────────────────────────────────────────
class PigModel(nn.Module):
    def __init__(self, backbone, dropout=0.3):
        super().__init__()
        if "dinov2" in backbone:
            self.encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
        else:
            self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
            
        nf = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(nf, 256), 
            nn.ReLU(),
            nn.Dropout(dropout/2), 
            nn.Linear(256, 1)
        )
    def forward(self, x): 
        return self.head(self.encoder(x)).squeeze(1)

# ─────────────────────────────────────────────────────────────
# 🖼️ Preprocessing
# ─────────────────────────────────────────────────────────────
def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2()
    ])

def predict(model, image_path, transform):
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor_img = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
            pred = model(tensor_img).item()
    return pred

# ─────────────────────────────────────────────────────────────
# 🏁 Benchmark Execution
# ─────────────────────────────────────────────────────────────
def main():
    print(f"--- Benchmarking Image: {os.path.basename(IMAGE_PATH)} ---")
    print("Pre-processing: Using YOLO segmentation to isolate pig...")
    
    yolo_model = YOLO(r"..\bestv8_seg.pt")
    results_yolo = yolo_model(IMAGE_PATH, conf=0.01, verbose=False)
    
    if not results_yolo[0].masks:
        print("Error: YOLO could not segment any pig. Performance will be degraded.")
        # Fallback to original image
        processed_img_np = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    else:
        # ใช้ Mask ตัวแรกที่เจอ
        mask = results_yolo[0].masks.data[0].cpu().numpy()
        orig_img = cv2.imread(IMAGE_PATH)
        h, w = orig_img.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        black_bg = np.zeros_like(orig_img)
        black_bg[mask_resized > 0] = orig_img[mask_resized > 0]
        
        # Crop
        box = results_yolo[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cropped = black_bg[y1:y2, x1:x2]
        processed_img_np = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        print("Success: Pig isolated on black background.")

    print("-" * 80)
    print(f"{'Model Architecture / Fold':<50} | {'Predicted Weight':<15}")
    print("-" * 80)

    # ค้นหาโมเดลทั้งหมด
    model_paths = glob(os.path.join(OUTPUT_DIR, "**", "*.pth"), recursive=True)
    
    results = []

    for path in sorted(model_paths):
        # เดา backbone จากชื่อโฟลเดอร์
        backbone = "convnext_small" # default
        img_size = 224
        
        if "dinov2" in path.lower():
            backbone = "dinov2_small"
            img_size = 518
        elif "efficientnet_b3" in path.lower():
            backbone = "efficientnet_b3"
            img_size = 300
        elif "efficientnet_b5" in path.lower():
            backbone = "efficientnet_b5"
            img_size = 456
        
        # แสดงชื่อที่เป็นมิตร
        rel_path = os.path.relpath(path, OUTPUT_DIR)
        
        try:
            # โหลดโมเดล
            model = PigModel(backbone).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            # โหลดแบบไม่เคร่งครัดสำหรับ DINOv2 mask_token
            model.load_state_dict(state_dict, strict=False)
            
            # ทำนาย
            tf = get_transform(img_size)
            tensor_img = tf(image=processed_img_np)["image"].unsqueeze(0).to(DEVICE)
            
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                    prediction = model(tensor_img).item()
            
            print(f"{rel_path:<50} | {prediction:>12.2f} kg")
            results.append((rel_path, prediction))
            
        except Exception as e:
            print(f"{rel_path:<50} | ERROR: {str(e)[:40]}")

    print("-" * 80)
    if results:
        avg_weight = sum(p for r, p in results) / len(results)
        print(f"{'ENSEMBLE AVERAGE':<50} | {avg_weight:>12.2f} kg")

if __name__ == "__main__":
    main()
