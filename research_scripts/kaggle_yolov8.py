"""
Kaggle Training Script for YOLOv8 (Pig Detection)
วิธีใช้:
1. อัปโหลดโค้ดนี้ลงใน Kaggle Notebook (copy-paste ไปใส่ Cell)
2. เพิ่ม Dataset (โฟลเดอร์รูปหมูแบบ YOLO) เข้าไปใน Notebook ฝั่งขวามือ (Add Input)
3. เปลี่ยนชื่อตัวแปร `input_dataset_dir` ด้านล่างให้ชี้ไปที่โฟลเดอร์ Dataset บน Kaggle
4. เปิด Internet ใน Settings ของ Notebook
5. กด Run All ยิงยาว 100 epochs!
"""

import os
import yaml
from ultralytics import YOLO

def prepare_data_yaml(input_yaml_path, output_yaml_path, root_dir):
    """อ่านและสร้างไฟล์ data.yaml ใหม่ให้ชี้ Path ไปที่ Dataset บน Kaggle ถูกต้อง"""
    with open(input_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # บังคับอัปเดต Path ให้เป็น Absolute Path ของ Kaggle
    data['train'] = os.path.join(root_dir, 'train', 'images')
    data['val'] = os.path.join(root_dir, 'valid', 'images')
    if 'test' in data:
        data['test'] = os.path.join(root_dir, 'test', 'images')
        
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"✅ สร้างไฟล์ Config เรียบร้อยที่: {output_yaml_path}")
    print(f"   Train path: {data['train']}")
    print(f"   Val path:   {data['val']}")

def main():
    # 1. ติดตั้งไลบรารี
    print("📦 กำลังติดตั้ง Ultralytics...")
    os.system('pip install ultralytics -q')
    
    print("🚀 เริ่มกระบวนการเตรียมเทรน YOLOv8 บน Kaggle...")
    
    # 📌 2. ระบุที่อยู่ของ Dataset ที่อัปโหลดใน Kaggle (แก้ตรงนี้ให้ตรงกับชื่อโฟลเดอร์)
    # เช่น '/kaggle/input/my-pig-yolo-dataset'
    input_dataset_dir = '/kaggle/input/PUT_YOUR_DATASET_NAME_HERE' 
    
    # 3. สร้าง data.yaml ใหม่ไว้ใน /kaggle/working เพื่อสิทธิ์ในการอ่าน/เขียน
    working_dir = '/kaggle/working'
    original_yaml = os.path.join(input_dataset_dir, 'data.yaml')
    new_yaml = os.path.join(working_dir, 'data.yaml')
    
    if os.path.exists(original_yaml):
        prepare_data_yaml(original_yaml, new_yaml, input_dataset_dir)
    else:
        print(f"\n❌ ERROR: ห้ามลืมเปลี่ยนบรรทัด `input_dataset_dir` ด้านบน")
        print(f"ให้ชี้ไปที่โฟลเดอร์ dataset ของตัวเองใน Kaggle นะครับ!")
        print(f"ปัจจุบันหาไฟล์ไม่เจอที่: {original_yaml}")
        return

    # 4. โหลดโมเดล YOLOv8n (Nano) ตัวเริ่มต้น
    model = YOLO('yolov8n.pt')
    
    # 5. เริ่มเทรน! (ปิด early stopping, รัน 100 รอบเต็ม)
    print("\n🔥 เริ่ม Training Loop...")
    results = model.train(
        data=new_yaml,
        epochs=100,               # รันไปเลย 100 รอบเต็ม
        patience=0,               # สำคัญมาก: ปิด Early Stopping ให้มันขยันเรียนจนจบ
        imgsz=640,
        batch=16,                 # ทรัพยากร Kaggle (T4 x2) อัด batch 16 สบายๆ
        name='yolov8_custom_pig_kaggle',
        project='/kaggle/working/outputs',
        device=0,                 # ใช้ GPU
        verbose=True
    )
    
    print("\n✅ การเทรนเสร็จสมบูรณ์!")
    print("👇 ดาวน์โหลดไฟล์โมเดลที่ดีที่สุด (best.pt) กลับไปเครื่องตัวเองได้เลยที่มุมขวา (Output):")
    print("/kaggle/working/outputs/yolov8_custom_pig_kaggle/weights/best.pt")

if __name__ == '__main__':
    main()
