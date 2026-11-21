import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import timm

# ─────────────────────────────────────────────────────────────
# 🛠️ Configuration
# ─────────────────────────────────────────────────────────────
MODEL_PATH = r"..\outputs\checkpoints\convnext_small_Sec\best_fold1.pth"
BACKBONE = "dinov2_small" # เปลี่ยนเป็น DINOv2
IMAGE_PATH = r"..\test_images\farm_test_22.jpg" # รูปที่บอสถ่ายมา
ACTUAL_WEIGHT = None # ไม่มีเฉลยน้ำหนักจริงให้เทียบ

# ─────────────────────────────────────────────────────────────
# 🐷 Model Architecture (ต้องเหมือนตอนเทรนเป๊ะๆ)
# ─────────────────────────────────────────────────────────────
class PigModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        # ใช้ pretrained=False ได้เพราะเราจะโหลด weight ทับอยู่แล้ว
        # dinov2_vits14 เป็นชื่อที่ถูกต้องใน timm สำหรับ DINOv2-Small ที่เราใช้เทรน
        self.encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0) # vit_small_patch14_dinov2.lvd142m คือชื่อของ dinov2_small ใน timm (ถ้าใช้ dinov2_vits14 ตอนเทรน ให้แก้ตรงนี้)
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
# 🖼️ Image Preprocessing (ต้องเป๊ะเหมือน validation_fold)
# ─────────────────────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
val_tf = A.Compose([
    A.Resize(518, 518), 
    A.Normalize(mean=MEAN, std=STD), 
    ToTensorV2()
])

def predict_weight(img_path, model_path):
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"[ERROR] ไม่พบไฟล์โมเดล '{model_path}'")
    if not os.path.exists(img_path):
         raise FileNotFoundError(f"[ERROR] ไม่พบไฟล์รูปภาพ '{img_path}'")

    print(f"[INFO] Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. โหลดโครงสร้าง
    model = PigModel().to(device)
    
    # 2. โลดยัดน้ำหนักที่ฝึกมา (จัดการกรณี torch.compile "_orig_mod" ด้วย)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # ลบ prefix '_orig_mod.' ถ้าฝั่ง HPC เคยรัน torch.compile แล้วติดมา
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()

    # 3. เตรียมรูป
    print(f"[INFO] Processing image: {img_path}")
    image = np.array(Image.open(img_path).convert("RGB"))
    tensor_img = val_tf(image=image)["image"].unsqueeze(0).to(device)
    
    # 4. ทำนาย
    print("[INFO] Predicting...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            prediction = model(tensor_img).item()

    return prediction

# ─────────────────────────────────────────────────────────────
# 🏁 Run Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        predicted = predict_weight(IMAGE_PATH, MODEL_PATH)
        print("\n" + "="*50)
        if ACTUAL_WEIGHT is not None:
            print(f"[RESULT] Actual Weight:     {ACTUAL_WEIGHT:.2f} kg")
            print(f"[RESULT] Predicted Weight:  {predicted:.2f} kg")
            print(f"[RESULT] Error margin:      {abs(predicted - ACTUAL_WEIGHT):.2f} kg")
        else:
            print(f"[RESULT] Predicted Weight:  {predicted:.2f} kg")
            print(f"[RESULT] Actual Weight:     (Not Provided)")
        print("="*50)
    except Exception as e:
        print(f"\n[ERROR] {e}")
