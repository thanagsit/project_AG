import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import sys
import os

# Setup path to import project modules
ROOT = Path("p:/projectAG_New")
sys.path.insert(0, str(ROOT))
from models import build_model
from data import get_val_transforms, PigDataset
from utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = ROOT / "PigRGB_Weight/RGB_9579"
CKPT_ROOT = ROOT / "outputs/checkpoints"

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def get_predictions(model_name, backbone, img_size, ckpt_dir, subsample_per_fold=100):
    print(f"Processing {model_name}...")
    results = {"true": [], "pred": []}
    
    transform = get_val_transforms(img_size)
    
    # Mock cfg for build_model
    cfg = OmegaConf.create({
        "model": {
            "backbone": backbone,
            "pretrained": False,
            "dropout": 0.3
        }
    })
    
    for fold in range(1, 6):
        ckpt_path = ckpt_dir / f"best_fold{fold}.pth"
        if not ckpt_path.exists():
            print(f"  Fold {fold} checkpoint missing.")
            continue
            
        # Build dataset for this validation fold
        val_fold_dir = [str(DATA_ROOT / f"fold{fold}")]
        dataset = PigDataset(fold_dirs=val_fold_dir, transform=transform)
        
        # Subsample to keep it fast
        indices = np.random.choice(len(dataset), min(subsample_per_fold, len(dataset)), replace=False)
        subset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, num_workers=0, 
            sampler=torch.utils.data.SubsetRandomSampler(indices)
        )
        
        # Load model
        model = build_model(cfg)
        checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)
        state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
        model.load_state_dict(checkpoint[state_key], strict=False)
        model.to(DEVICE)
        model.eval()
        
        with torch.no_grad():
            for imgs, labels in subset_loader:
                imgs = imgs.to(DEVICE)
                preds = model(imgs)
                results["true"].extend(labels.cpu().numpy())
                results["pred"].extend(preds.cpu().numpy())
        
        print(f"  Fold {fold} done.")
        
    return results

# Model configurations
models_to_plot = [
    {
        "name": "ConvNeXt Small",
        "backbone": "convnext_small",
        "img_size": 224,
        "ckpt_dir": CKPT_ROOT / "convnext_small",
        "color": "#FF9999" # Pastel Red
    },
    {
        "name": "EfficientNet B5",
        "backbone": "efficientnet_b5",
        "img_size": 224,
        "ckpt_dir": CKPT_ROOT / "efficientnet_b5",
        "color": "#99FF99" # Pastel Green
    },
    {
        "name": "DINOv2 Small",
        "backbone": "vit_small_patch14_dinov2.lvd142m",
        "img_size": 518,
        "ckpt_dir": CKPT_ROOT / "dinov2_small",
        "color": "#9999FF" # Pastel Blue
    }
]

plt.figure(figsize=(10, 8))

all_data = []
for m_cfg in models_to_plot:
    data = get_predictions(
        m_cfg["name"], 
        m_cfg["backbone"], 
        m_cfg["img_size"], 
        m_cfg["ckpt_dir"],
        subsample_per_fold=200 # Total ~1000 points per model
    )
    plt.scatter(
        data["true"], data["pred"], 
        alpha=0.5, s=20, label=m_cfg["name"], color=m_cfg["color"], edgecolors='w', linewidth=0.5
    )
    all_data.extend(data["true"])
    all_data.extend(data["pred"])

# Add identity line
min_val = min(all_data) - 5
max_val = max(all_data) + 5
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label="Ideal")

plt.xlabel("True Values (kg)", fontdict={'fontsize': 12, 'fontweight': 'bold'})
plt.ylabel("Predicted Values (kg)", fontdict={'fontsize': 12, 'fontweight': 'bold'})
plt.title("Model Prediction vs True Values Correlation", fontdict={'fontsize': 14, 'fontweight': 'bold'}, pad=20)
plt.legend(title="Model Architectures", title_fontsize='11', fontsize='10')
plt.grid(True, linestyle='--', alpha=0.6)

# Tight layout
plt.tight_layout()

# Save plot
plot_path = ROOT / "research_scripts/regression_plot.png"
plt.savefig(str(plot_path), dpi=300)
print(f"\nPlot saved to: {plot_path}")
