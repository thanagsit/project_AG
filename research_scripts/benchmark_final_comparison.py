import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup path to import project modules
ROOT = Path("p:/projectAG_New")
sys.path.insert(0, str(ROOT))
from models import build_model
from utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = ROOT / "test_images/farm_test_10.jpg"

def get_tf(img_size):
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

def predict_weight(model, image_tensor, device):
    with torch.no_grad():
        batch = image_tensor.unsqueeze(0).to(device)
        pred = model(batch)
        return pred.item()

def run_benchmark(model_name, backbone, img_size, ckpt_dir):
    print(f"\n--- Benchmarking {model_name} ({backbone}) ---")
    
    # Mock cfg for build_model
    cfg = OmegaConf.create({
        "model": {
            "backbone": backbone,
            "pretrained": False,
            "dropout": 0.3
        }
    })
    
    transform = get_tf(img_size)
    img = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = transform(image=np.array(img))["image"]
    
    ckpt_path_list = sorted(list(ckpt_dir.glob("best_fold*.pth")))
    if not ckpt_path_list:
        print(f"❌ No checkpoints found in {ckpt_dir}")
        return None
    
    preds = []
    for i, ckpt_path in enumerate(ckpt_path_list):
        model = build_model(cfg)
        
        # Manually load to handle non-strict for DINOv2
        checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)
        state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
        model.load_state_dict(checkpoint[state_key], strict=False)
        
        model.to(DEVICE)
        model.eval()
        
        weight = predict_weight(model, image_tensor, DEVICE)
        preds.append(weight)
        print(f"  Fold {i+1}: {weight:.2f} kg")
        
    avg_weight = sum(preds) / len(preds)
    print(f"Average: {avg_weight:.2f} kg")
    return avg_weight

results = {}

# 1. ConvNeXt Small
results["ConvNeXt Small"] = run_benchmark(
    "ConvNeXt Small", 
    "convnext_small", 
    224, 
    ROOT / "outputs/checkpoints/convnext_small"
)

# 2. EfficientNet B5
results["EfficientNet B5"] = run_benchmark(
    "EfficientNet B5", 
    "efficientnet_b5", 
    224, 
    ROOT / "outputs/checkpoints/efficientnet_b5"
)

# 3. DINOv2 Small
results["DINOv2 Small"] = run_benchmark(
    "DINOv2 Small", 
    "vit_small_patch14_dinov2.lvd142m", 
    518, 
    ROOT / "outputs/checkpoints/dinov2_small"
)

print("\n" + "="*30)
print("FINAL COMPARISON SUMMARY")
print("="*30)
for name, val in results.items():
    if val:
        print(f"{name:<20}: {val:.2f} kg")
