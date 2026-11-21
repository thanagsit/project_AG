"""
scripts/test_yolo_custom.py
สคริปต์สั้นๆ ไว้รีวิวว่า YOLOv8 ที่สอนใหม่ (Pig-specific) ตีกรอบแม่นขนาดไหน
"""

from ultralytics import YOLO
from pathlib import Path

def test_custom_yolo():
    # โหลด YOLOv8 โมเดลจาก Kaggle
    model_path = Path("p:/projectAG_New/bestv8_last.pt")
    if not model_path.exists():
        print("❌ ไม่พบไฟล์โมเดลที่:", model_path)
        return

    model = YOLO(model_path)
    print("✅ โหลดโมเดล Custom Pig แสนรู้สำเร็จ!")

    # สุ่มเลือกภาพมาทดสอบ
    test_image = "p:/projectAG_New/test_images/farm_test_01.jpg"
    print(f"🔍 สแกนหมูในภาพ: {test_image}")

    # รัน Predict (save=True จะบันทึกรูปผลลัพธ์ที่ตีกรอบแล้ว)
    results = model.predict(source=test_image, save=True, conf=0.10)
    
    print("\n✅ การทดสอบเสร็จสิ้น!")
    save_dir = results[0].save_dir
    print(f"👉 ภาพผลลัพธ์ (พร้อมกรอบเขียวๆ) บันทึกอยู่ที่:\n   {save_dir}")

if __name__ == "__main__":
    test_custom_yolo()
