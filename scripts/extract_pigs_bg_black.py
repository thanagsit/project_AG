"""
สคริปต์สกัดรูปร่างหมูแบบลบฉากหลัง (Background Removal via YOLO-Seg)
สคริปต์นี้เอาไว้เตรียม Dataset ขั้นท็อปสุด สำหรับให้ DINOv2 หรือ ConvNeXt เทรน

ขั้นตอนการทำงาน:
1. โหลดโมเดล YOLOv8-Seg (ตัดขอบ)
2. สแกนรูปถ่ายหมูทั้งหมดใน Dataset ต้นฉบับ (9,500+ รูป)
3. ยิง YOLO-Seg ให้มันบอกตำแหน่งตีกรอบ (กล่อง) และหน้ากาก (Mask)
4. เลื่อยตัด (Crop) ภาพตามกรอบกล่อง
5. หยด "สีดำสนิท" ลงไปในทุกพิกเซลที่ Mask บอกว่า "ไม่ใช่เนื้อหมู" (ลบพื้นหลัง)
6. บันทึกรูปหมูลอยได้ในอวกาศพื้นดำ ลงใน Dataset โฟลเดอร์ใหม่ (โครงสร้างน้ำหนักเดิม)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# ⚙️ ตั้งค่า (แก้ไขให้ตรงกับเครื่องของนาย)
# ==========================================

# พาธโมเดล YOLOv8-Seg ที่เทรนเสร็จแล้วจาก Kaggle (เอากลับมาลงเครื่อง)
YOLO_SEG_WEIGHTS = "bestv8_seg.pt" 

# โฟลเดอร์ต้นฉบับ ที่เก็บรูปเรียงตามน้ำหนัก (รูปเต็มใบ)
SOURCE_DIR = r"p:\projectAG_New\PigRGB_Weight\RGB_9579"

# โฟลเดอร์ปลายทาง ที่จะเซฟรูปพื้นหลังดำ (จัดโครงสร้างน้ำหนักเหมือนเดิมเป๊ะ)
DEST_DIR = r"p:\projectAG_New\PigRGB_BlackBg_9579"

# ==========================================

def process_dataset_to_black_bg():
    if not os.path.exists(YOLO_SEG_WEIGHTS):
        print(f"❌ ไม่พบโมเดล YOLO-Seg ที่พาธ: {YOLO_SEG_WEIGHTS}")
        print("💡 รบกวนดาวน์โหลด best.pt จาก Kaggle มาเปลี่ยนชื่อเป็น bestv8_seg.pt วางไว้ในโปรเจคก่อนครับ")
        return

    print("🧠 กำลังโหลดสมองกล YOLO-Seg...")
    model = YOLO(YOLO_SEG_WEIGHTS)
    
    src_path = Path(SOURCE_DIR)
    dst_path = Path(DEST_DIR)
    
    # ดึงรูปทั้งหมดข้างในโดยเข้าไปลึกทุก Sub-folder (เช่น ทะลุ fold1, fold2 เข้าไปหาโฟลเดอร์น้ำหนัก)
    img_files = list(src_path.rglob("*.jpg")) + list(src_path.rglob("*.png"))
    print(f"📂 พบรูปภาพรอการสกัดทั้งหมด: {len(img_files)} รูป")
    
    total_images_processed = 0
    total_images_failed = 0
    
    for img_file in tqdm(img_files, desc="Extracting Black-Bg Pigs"):
        
        # หาสายเลือดของมันแต่เดิม (relative path ภายใต้ SOURCE_DIR)
        relative_path = img_file.relative_to(src_path)
        
        # สร้างโฟลเดอร์รอรับที่ปลายทาง ให้ล้อตามโฟลเดอร์แหล่งกำเนิดเป๊ะๆ
        out_file = dst_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
            
        # ถ้าเคยทำแล้วข้ามได้เลย (Resume capability)
        if out_file.exists():
            total_images_processed += 1
            continue
            
        # โหลดรูปต้นฉบับด้วย OpenCV
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        # ยิง YOLO-Seg สแกนหาหมู
        # (ตั้งค่า conf=0.5 ลดลงหน่อยให้มันใจกล้าหาให้เจอ)
        results = model.predict(img, conf=0.5, verbose=False)
        
        if len(results) == 0 or results[0].masks is None:
            # ถ้า YOLO หาหมูในรูปนี้ไม่เจอเลย (อาจจะภาพมืดไป ขาเดี่ยวไป)
            total_images_failed += 1
            continue
            
        result = results[0]
        
        # หาตัวหมูที่มันมั่นใจที่สุด (ตัวที่มีค่า confidence สูงสุด)
        best_idx = np.argmax(result.boxes.conf.cpu().numpy())
        
        # 1. ดึงกรอบสี่เหลี่ยม (x1, y1, x2, y2)
        box = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        
        # กันเหนียวเผื่อขอบมันทะลุภาพ
        h, w = img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        # 2. ดึงหน้ากากขาวดำ (Mask)
        # YOLO ส่งกลับมาเป็น array สัดส่วนเท่ารูปจริง ค่า 0.0-1.0
        mask = result.masks.data[best_idx].cpu().numpy()
        
        # Resize mask ให้พิกเซลตรงกับรูปจริงเป๊ะๆ ในกรณีที่ YOLO บีบอัดไปคิด
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 3. สร้างรูปผลลัพธ์ที่เป็นสีดำล้วนๆ ก่อน
        black_bg_img = np.zeros_like(img)
        
        # 4. ระบายสีเนื้อหมูจากรูปดั้งเดิม ลงไปบนรูปสีดำ (เฉพาะจุดที่ mask โดนหมู)
        # เอา mask 2 มิติแปลเป็น 3 มิติ (RGB)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # ตรงไหน mask เป็น 1 (มีหมู) จะดึงสีจริงมา, ตรงไหนเป็น 0 จะเอาจาก black_bg_img (สีดำ)
        masked_full_img = np.where(mask_3d > 0.5, img, black_bg_img)
        
        # 5. Crop ตัดเฉพาะรูปหมู เอาดินสอขีดเส้นสี่เหลี่ยมออกมา
        final_crop = masked_full_img[y1:y2, x1:x2]
        
        # 6. บันทึกรูปภาพ
        cv2.imwrite(str(out_file), final_crop)
        total_images_processed += 1
            
    print("\n🎉 สกัดพื้นหลังดำเสร็จสมบูรณ์!")
    print(f"👉 รูปผ่านฉลุย: {total_images_processed} รูป")
    print(f"❌ รูปที่ YOLO หาหมูไม่พบ: {total_images_failed} รูป")
    print(f"📂 ไฟล์ทั้งหมดถูกเก็บไว้ที่: {DEST_DIR}")

if __name__ == '__main__':
    process_dataset_to_black_bg()
