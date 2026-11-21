"""
Kaggle Script: YOLOv8 Instance Segmentation (YOLO-Seg)
สคริปต์นี้เอาไว้แปลงรูป "Mask (ขาว-ดำ)" ให้กลายเป็น "Polygon (จุดพิกัด)" แบบที่ YOLO-Seg ต้องการ
และสั่งเทรนโมเดล YOLOv8-Seg ต่อทันที!

วิธีใช้:
1. นำโฟลเดอร์รูปและโฟลเดอร์ Mask ขึ้น Kaggle
2. แก้ไขพาธ dataset ตรงตั้งค่า
3. รันโค้ดยาวๆ ได้เลย
"""

import os
import sys

# 📦 ติดตั้งไลบรารีที่จำเป็นก่อน Import เสมอ (สำหรับ Kaggle)
print("📦 กำลังติดตั้ง Ultralytics และไลบรารีที่เกี่ยวข้อง...")
os.system('pip install ultralytics scikit-learn opencv-python-headless -q')

import cv2
import yaml
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================================
# ⚙️ 1. ตั้งค่า (แก้ตรงนี้ให้ตรงกับโฟลเดอร์ใน Kaggle)
# ==========================================================
# พาธของ Dataset บน Kaggle
KAGGLE_INPUT_BASE = "/kaggle/input/datasets/ppp123zz/pig-rgb-weight"

_candidates = [
    os.path.join(KAGGLE_INPUT_BASE, "RGB_MASK_3394"),
    os.path.join(KAGGLE_INPUT_BASE, "PigRGB_Weight", "RGB_MASK_3394"),
    os.path.join(KAGGLE_INPUT_BASE, "PigRGB_Weight"),
    KAGGLE_INPUT_BASE,
]
KAGGLE_INPUT_DIR = next((p for p in _candidates if os.path.exists(p) and "RGB_MASK_3394" in p), KAGGLE_INPUT_BASE)
print(f"✅ ค้นพบโฟลเดอร์ Mask ที่: {KAGGLE_INPUT_DIR}")

# โฟลเดอร์ปลายทางที่จะสร้างชุดข้อมูลแบบ YOLO-Seg
YOLO_SEG_DATASET_DIR = "/kaggle/working/yolo_seg_dataset"

IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16

def create_yolo_seg_dataset():
    """ฟังก์ชันแปลงรูปภาพ Mask สีขาว-ดำ ให้ออกมาเป็นไฟล์ .txt (Polygon X Y)"""
    print("🚀 กำลังเตรียม Dataset สำหรับ YOLO-Seg...")
    
    img_dir_source = Path(KAGGLE_INPUT_DIR) / "RGB_3394"
    mask_dir_source = Path(KAGGLE_INPUT_DIR) / "MASK_3394"
    
    if not img_dir_source.exists() or not mask_dir_source.exists():
        print(f"❌ หาโฟลเดอร์ไม่เจอ! เช็คพาธ KAGGLE_INPUT_DIR ให้ดีๆ นะครับ")
        print(f"👉 คาดหวังว่าจะมี: {img_dir_source}")
        return False

    # สร้างโฟลเดอร์ปลายทาง
    for split in ['train', 'valid']:
        (Path(YOLO_SEG_DATASET_DIR) / split / 'images').mkdir(parents=True, exist_ok=True)
        (Path(YOLO_SEG_DATASET_DIR) / split / 'labels').mkdir(parents=True, exist_ok=True)

    # รวบรวมรูปทั้งหมด (วนลูปเข้าไปในแต่ละโฟลเดอร์น้ำหนักย่อย)
    all_pairs = []
    for weight_folder in img_dir_source.iterdir():
        if not weight_folder.is_dir(): continue
        
        # ค้นหาทั้ง .jpg และ .png ในโฟลเดอร์ภาพจริง
        for img_path in list(weight_folder.glob("*.jpg")) + list(weight_folder.glob("*.png")):
            mask_name = img_path.name # ใช้ชื่อเดียวกันเป๊ะ ๆ เลย
            mask_path = mask_dir_source / weight_folder.name / mask_name
            
            # ยืดหยุ่น: บางทีรูปจริงเป็น .jpg แต่หน้ากากถูกเซฟเป็น .png
            if not mask_path.exists():
                mask_path = mask_dir_source / weight_folder.name / (img_path.stem + ".png")
                
            if mask_path.exists():
                all_pairs.append((img_path, mask_path))

    print(f"✅ พบภาพที่จับคู่กับ Mask สำเร็จ: {len(all_pairs)} คู่")
    if len(all_pairs) == 0:
        return False

    # สุ่มแบ่ง Train 80% / Val 20%
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    
    def process_split(pairs, split_name):
        print(f"⏳ กำลังแปลง {split_name} ({len(pairs)} รูป)...")
        for img_p, mask_p in tqdm(pairs):
            # 1. ก็อปปี้ภาพจริงไปโฟลเดอร์ images
            out_img = Path(YOLO_SEG_DATASET_DIR) / split_name / 'images' / img_p.name
            shutil.copy(img_p, out_img)
            
            # 2. อ่านภาพ Mask แปลงเป็นขาวดำ
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            
            # แปลงเป็น Binary (หมู=สีขาว พื้นหลัง=สีดำ)
            _, thresholded = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # 3. หากรอบ Polygon (จุดไข่ปลารอบตัวหมู)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 4. เขียนไฟล์ .txt
            label_name = img_p.stem + ".txt"
            out_txt = Path(YOLO_SEG_DATASET_DIR) / split_name / 'labels' / label_name
            
            with open(out_txt, 'w') as f:
                for contour in contours:
                    # ถ้าเส้นเล็กเกินไป (Noise) ให้ข้าม
                    if cv2.contourArea(contour) < 500: continue
                        
                    # ย่อพิกัดให้อยู่ในสเกล 0.0 - 1.0 ตามกฎของ YOLO
                    polygon = []
                    for point in contour:
                        x = point[0][0] / w
                        y = point[0][1] / h
                        polygon.append(f"{x:.6f} {y:.6f}")
                        
                    # คลาส 0 = Pig
                    f.write(f"0 {' '.join(polygon)}\n")

    process_split(train_pairs, 'train')
    process_split(val_pairs, 'valid')
    
    # 5. สร้างไฟล์ data_seg.yaml
    yaml_dict = {
        'path': YOLO_SEG_DATASET_DIR,
        'train': 'train/images',
        'val': 'valid/images',
        'names': {0: 'pig'}
    }
    with open(f'{YOLO_SEG_DATASET_DIR}/data_seg.yaml', 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print(f"✨ สร้าง Dataset Polygon สำเร็จเรียบร้อย! {YOLO_SEG_DATASET_DIR}/data_seg.yaml")
    return True

# ==========================================================
# 🚀 2. ฟังก์ชันเริ่ม Train
# ==========================================================
import subprocess

def main():
    # ติดตั้งไลบรารี
    print("📦 กำลังติดตั้งสภาพแวดล้อม...")
    os.system('pip install ultralytics scikit-learn opencv-python-headless -q')
    
    # ดึง Data / สร้าง Polygon
    success = create_yolo_seg_dataset()
    if not success:
        return
        
    yaml_path = f"{YOLO_SEG_DATASET_DIR}/data_seg.yaml"
    
    print("\n🔥 เริ่มทำInstance Segmentation Training Loop แบบ Multi-GPU (DDP)...")
    
    # ใช้ Subprocess สั่งรันผ่าน CLI เพื่อทะลุกำแพง Notebook ปลุก T4 ใบที่สอง!
    cmd = [
        "yolo", "segment", "train", 
        f"data={yaml_path}", 
        "model=yolov8n-seg.pt", 
        f"epochs={EPOCHS}",
        "patience=25",
        f"imgsz={IMG_SIZE}",
        f"batch={BATCH_SIZE}",
        "name=yolov8_seg_pig",
        "project=/kaggle/working/outputs",
        "device=0,1" # 💥 เปิดโหมด Parallel เต็มสูบ (T4 x2)
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("\n✅ การเทรนเสร็จสมบูรณ์!")
    print("👇 ดาวน์โหลดไฟล์โมเดลที่ดีที่สุด (best.pt) กลับไปเครื่องตัวเองได้เลยที่มุมขวา (Output):")
    print("/kaggle/working/outputs/yolov8_seg_pig/weights/best.pt")

if __name__ == '__main__':
    main()
