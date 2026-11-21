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

# ─────────────────────────────────────────────────────────────
# 🛠️ Configuration
# ─────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = r"..\bestv8_seg.pt"  # โมเดล YOLOv8 Segmentation
DINOV2_MODEL_PATH = r"..\outputs\checkpoints\convnext_small_Sec\best_fold1.pth"
IMAGE_PATH = r"..\test_images\farm_test_01.jpg"
CONF_THRESHOLD = 0.1

# ─────────────────────────────────────────────────────────────
# 🐷 DINOv2 Architecture
# ─────────────────────────────────────────────────────────────
class PigModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
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
# 🖼️ Image Preprocessing (DINOv2)
# ─────────────────────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
val_tf = A.Compose([
    A.Resize(518, 518), 
    A.Normalize(mean=MEAN, std=STD), 
    ToTensorV2()
])

def load_dinov2(model_path, device):
    model = PigModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def process_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[INFO] Loading Models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    dinov2_model = load_dinov2(DINOV2_MODEL_PATH, device)

    print(f"[INFO] Processing image: {IMAGE_PATH}")
    # 1. รัน YOLO Segmentation
    results = yolo_model(IMAGE_PATH, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]
    
    if result.masks is None:
        print("[ERROR] No pigs detected or segmentation failed.")
        return

    orig_img_cv = cv2.imread(IMAGE_PATH)
    h, w = orig_img_cv.shape[:2]
    
    # วาดรูปลงบนภาพต้นฉบับ
    output_img_cv = orig_img_cv.copy()

    for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
        conf = box.conf[0].item()
        
        # --- สร้างภาพพื้นหลังดำ (Black Background) ---
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
        black_bg = np.zeros_like(orig_img_cv)
        black_bg[mask_resized > 0] = orig_img_cv[mask_resized > 0]

        # Crop ตาม Bounding Box ของหมูตัวนั้นๆ
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cropped_black_bg = black_bg[y1:y2, x1:x2]
        
        # ป้องกัน error กรณี crop ได้ภาพเล็กเกินไปหรือพิกัดผิด
        if cropped_black_bg.size == 0:
            continue

        # เปลี่ยนจาก BGR (OpenCV) เป็น RGB สำหรับเข้าโมเดล DINOv2
        cropped_rgb = cv2.cvtColor(cropped_black_bg, cv2.COLOR_BGR2RGB)
        
        # --- 2. ทายน้ำหนักด้วย DINOv2 ---
        tensor_img = val_tf(image=cropped_rgb)["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                predicted_weight = dinov2_model(tensor_img).item()

        print(f"\n[RESULT] Pig {i+1} (Conf: {conf:.2f}): Predicted Weight = {predicted_weight:.2f} kg")

        # --- 3. วาดผลลัพธ์ลงบนภาพ ---
        cv2.rectangle(output_img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        text = f"{predicted_weight:.1f} kg"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            
        cv2.rectangle(output_img_cv, (text_x, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(output_img_cv, text, (text_x + 2, text_y), font, font_scale, (0, 255, 0), thickness)

    # Save output
    os.makedirs(r"..\outputs\predictions", exist_ok=True)
    out_filename = os.path.basename(IMAGE_PATH)
    out_path = os.path.join(r"..\outputs\predictions", f"dinov2_seg_{out_filename}")
    cv2.imwrite(out_path, output_img_cv)
    print(f"\n[INFO] Saved visualization with segmented weight predictions to: {out_path}")

if __name__ == "__main__":
    process_and_predict()
