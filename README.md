# 🐷 Pig Weight Prediction

โมเดล Deep Learning สำหรับทำนายน้ำหนักหมูจากภาพถ่าย

## Project Structure

```
projectAG_New/
├── config/              # config.yaml (hyperparameters ทั้งหมด)
├── data/                # Dataset class, Transforms
├── models/              # EfficientNet backbone, Loss functions
├── training/            # Trainer, Metrics (MAE, RMSE, R²)
├── utils/               # Logger, Checkpoint
├── scripts/             # Entry points
│   ├── train.py         #   → เทรน 5-fold CV (local)
│   ├── evaluate.py      #   → ประเมินผล checkpoints
│   ├── predict.py       #   → ทำนายน้ำหนักจากรูปภาพ
│   └── fix.py           #   → ตรวจสอบ dataset integrity
├── notebooks/           # Kaggle Notebooks
│   ├── kaggle_train.py  #   → เทรนบน Kaggle T4 (single model)
│   └── kaggle_compare.py#   → เปรียบเทียบ 3 โมเดลบน Kaggle
├── outputs/             # checkpoints/ + logs/ (auto-created)
├── PigRGB_Weight/       # Dataset (RGB_9579 + RGB_MASK_3394)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt

# เทรน local (ใช้ GPU ถ้ามี)
python scripts/train.py

# ประเมินผล
python scripts/evaluate.py

# ทำนายน้ำหนักจากรูป
python scripts/predict.py --image path/to/pig.png
```

## Dataset Format

```
RGB_9579/
├── fold1/
│   └── 114.5_138/       # folder name = weight_pigID
│       └── 114.5kg_1.png
```
น้ำหนัก parse ได้จากชื่อโฟลเดอร์ เช่น `114.5_138` → 114.5 kg

## Target Metrics

| Metric | เป้าหมาย | ผลที่ได้ |
|--------|---------|---------|
| MAE    | < 5 kg  | **4.56 kg** ✅ |
| R²     | > 0.90  | **0.9641** ✅ |

## Kaggle Training

1. Copy `notebooks/kaggle_train.py` ไปวางใน Kaggle Notebook
2. เปิด GPU T4 x1 (single model) หรือ T4 x2 (compare 3 models: `kaggle_compare.py`)
3. รัน → Download `outputs/checkpoints/`


โมเดล Deep Learning สำหรับทำนายน้ำหนักหมูจากภาพถ่าย  
ใช้ **EfficientNet-B3** เป็น backbone + **5-Fold Cross Validation**

## โครงสร้าง Project

```
├── config/config.yaml       ← hyperparameters ทั้งหมด
├── data/                    ← Dataset + Augmentation
├── models/                  ← EfficientNet backbone + loss
├── training/                ← Trainer + metrics
├── utils/                   ← logger + checkpoint
├── scripts/
│   ├── train.py             ← entry: เทรนโมเดล 5-fold
│   ├── evaluate.py          ← ประเมิน checkpoint ที่บันทึกไว้
│   └── predict.py           ← inference รูปเดียว
└── outputs/
    ├── checkpoints/         ← best_fold1.pth ... best_fold5.pth
    └── logs/                ← training.log, metrics.csv
```

## Quick Start

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. เทรนโมเดล
```bash
python scripts/train.py
```

### 3. ทดสอบ 1 epoch (smoke test)
```bash
python scripts/train.py --debug
```

### 4. Evaluate model
```bash
python scripts/evaluate.py
```

### 5. ทำนายน้ำหนักจากรูปภาพ
```bash
python scripts/predict.py --image path/to/pig.png
```

## Dataset Structure

```
PigRGB_Weight/RGB_9579/
    fold1/  fold2/  fold3/  fold4/  fold5/
        └── <weight_kg>_<num_images>/
                └── <weight_kg>kg_<n>.png
```

**Label parsing:** ชื่อ folder `114.5_138` → น้ำหนัก = **114.5 kg**

## ปรับแต่ง Hyperparameters

แก้ไขที่ `config/config.yaml` เท่านั้น — ไม่ต้องแตะโค้ด

```yaml
training:
  epochs: 50
  batch_size: 32
  lr: 1.0e-4
  patience: 10    # early stopping
  loss: "huber"   # huber | mae | mse
```
