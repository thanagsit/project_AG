import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Setup path to import project modules
ROOT = Path("p:/projectAG_New")
sys.path.insert(0, str(ROOT))
from models import build_model
from data import get_val_transforms
from utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_MODEL_DIR = ROOT / "outputs/checkpoints/convnext_small"
CONFIG_PATH = ROOT / "config/config.yaml"

# Sample images from training set
SAMPLES = [
    {
        "path": ROOT / "PigRGB_Weight/RGB_9579/fold1/73.36_124/73.36kg_1.png",
        "gt": 73.36
    },
    {
        "path": ROOT / "PigRGB_Weight/RGB_9579/fold1/112.7_114/112.7kg_1.png",
        "gt": 112.70
    },
    {
        "path": ROOT / "PigRGB_Weight/RGB_9579/fold1/186.26_121/186.26kg_1.png",
        "gt": 186.26
    }
]

cfg = OmegaConf.load(CONFIG_PATH)
cfg.model.backbone = "convnext_small" 
transform = get_val_transforms(cfg.data.img_size)

def predict_weight(model, image_tensor, device):
    with torch.no_grad():
        batch = image_tensor.unsqueeze(0).to(device)
        pred = model(batch)
        return pred.item()

print(f"Loading ConvNeXt Ensemble from {WEIGHT_MODEL_DIR}...")
weight_models = []
for fold_path in sorted(list(WEIGHT_MODEL_DIR.glob("best_fold*.pth"))):
    w_model = build_model(cfg)
    load_checkpoint(w_model, str(fold_path), device=DEVICE)
    w_model.to(DEVICE)
    w_model.eval()
    weight_models.append(w_model)

print("\n" + "="*60)
print(f"{'Image Path':<25} | {'Ground Truth':<12} | {'Predicted':<12} | {'Diff':<8}")
print("-" * 60)

for sample in SAMPLES:
    img = Image.open(sample["path"]).convert("RGB")
    image_tensor = transform(image=np.array(img))["image"]
    
    preds = [predict_weight(m, image_tensor, DEVICE) for m in weight_models]
    avg_weight = sum(preds) / len(preds)
    diff = avg_weight - sample["gt"]
    
    name = sample["path"].name
    print(f"{name:<25} | {sample['gt']:12.2f} | {avg_weight:12.2f} | {diff:8.2f}")
