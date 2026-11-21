import os
import cv2
import numpy as np

# ระบุโฟลเดอร์ในเครื่องของคุณ
dataset_path = r"P:\projectAG_New\PigRGB_Weight"

def validate_dataset(base_path):
    print(f"🚀 เริ่มต้นสแกนหาจุดบอด Dataset ที่: {base_path}\n")

    stats = {
        "fixed_spaces": 0,
        "rgb_count": 0,
        "mask_count": 0,
        "kg_missing": 0,
        "corrupted": 0,
        "empty_mask": 0
    }

    # 1. Path & Naming: ซ่อมแซมช่องว่าง (Trailing spaces) ก่อนเลย
    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in dirs + files:
            clean_name = name.strip()
            if clean_name != name:
                try:
                    os.rename(os.path.join(root, name), os.path.join(root, clean_name))
                    stats["fixed_spaces"] += 1
                except Exception:
                    pass

    # 2. เริ่มสแกนเจาะลึกทีละไฟล์
    print("⏳ กำลังใช้ cv2 เอ็กซเรย์รูปภาพทุกใบ (อาจใช้เวลาสักครู่)...")
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            if not filename.lower().endswith('.png'):
                continue

            filepath = os.path.join(root, filename)
            is_mask_folder = 'MASK' in root.upper()

            # เช็คชื่อไฟล์ว่ามีหน่วย kg ไหม
            if 'kg' not in filename.lower():
                stats["kg_missing"] += 1

            # นับแยกหมวดหมู่
            if is_mask_folder:
                stats["mask_count"] += 1
            else:
                stats["rgb_count"] += 1

            # 3. Corruption Check: เช็ครูปเสีย
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                stats["corrupted"] += 1
                print(f"❌ เจอรูปเสีย (เปิดไม่ได้): {filepath}")
                continue

            # 4. Mask Validity: เช็คว่าภาพ Mask ดำสนิทไหม
            if is_mask_folder:
                # ถ้าภาพอ่านมาแบบ RGB ให้แปลงเป็นขาวดำก่อนเช็ค
                if len(img.shape) == 3: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # ถ้านับพิกเซลสีขาวได้ 0 แปลว่าภาพนี้ไม่มีตัวหมู (ดำสนิท)
                if np.sum(img > 0) == 0:
                    stats["empty_mask"] += 1
                    print(f"⚠️ ภาพ Mask ไม่มีตัวหมู (ดำสนิท): {filepath}")

    # สรุปผลลัพธ์แบบ Report
    print("\n" + "="*50)
    print(" 📊 รายงานสถานะ Dataset (Dataset Health Report)")
    print("="*50)
    print(f"🔧 ซ่อมแซมชื่อโฟลเดอร์/ไฟล์เว้นวรรค: {stats['fixed_spaces']} จุด")
    print(f"📸 จำนวนรูปภาพ RGB ต้นฉบับ: {stats['rgb_count']} รูป")
    print(f"⚫ จำนวนรูปภาพ Mask (เฉลย): {stats['mask_count']} รูป")
    print("-" * 50)
    print(f"⚠️ ไฟล์ชื่อเพี้ยน (ไม่มี kg): {stats['kg_missing']} ไฟล์")
    print(f"❌ ไฟล์รูปภาพพัง (Corrupted): {stats['corrupted']} ไฟล์")
    print(f"👻 ไฟล์ Mask ดำสนิท (ใช้ไม่ได้): {stats['empty_mask']} ไฟล์")
    print("="*50)

    if stats['corrupted'] == 0 and stats['empty_mask'] == 0 and stats['rgb_count'] > 0:
        print("✅ ยอดเยี่ยม! Dataset คลีน 100% พร้อมลุยทำสมการแล้วครับ!")
    else:
        print("⚠️ มีไฟล์ที่อาจจะทำโมเดล Error ซ่อนอยู่! ลองเช็คไฟล์ตามรายชื่อด้านบนครับ")

# กดรันเลย!
validate_dataset(dataset_path)