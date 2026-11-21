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
MODEL_PATH = "../outputs/checkpoints/convnext_small_hpc/best_fold4.pth" # ใช้โมเดล HPC เดิมที่มีอยู่แล้ว
BACKBONE = "convnext_small"
IMAGE_PATH = r"..\PigRGB_Weight\RGB_9579\fold1\101.74_127\101.74kg_1.png" # ลองเปลี่ยนเป็นรูปอื่นดูได้
ACTUAL_WEIGHT = 101.74 # อิงตามชื่อโฟลเดอร์ (น้ำหนัก 101.74 kg)

# ─────────────────────────────────────────────────────────────
# 🐷 Model Architecture (ต้องเหมือนตอนเทรนเป๊ะๆ)
# ─────────────────────────────────────────────────────────────
class PigModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        # ใช้ pretrained=False ได้เพราะเราจะโหลด weight ทับอยู่แล้ว
        self.encoder = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool="avg")
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
    A.Resize(224, 224), 
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
    model.load_state_dict(state_dict)
    
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
        print(f"[RESULT] Actual Weight:     {ACTUAL_WEIGHT:.2f} kg")
        print(f"[RESULT] Predicted Weight:  {predicted:.2f} kg")
        print(f"[RESULT] Error margin:      {abs(predicted - ACTUAL_WEIGHT):.2f} kg")
        print("="*50)
    except Exception as e:
        print(f"\n[ERROR] {e}")
