import torch
from pathlib import Path
import json
import numpy as np

CKPT_ROOT = Path("p:/projectAG_New/outputs/checkpoints")
MODELS = ["convnext_small", "efficientnet_b5", "dinov2_small"]

def extract_metrics(model_dir):
    print(f"\nExtracting metrics from {model_dir.name}...")
    fold_results = []
    
    ckpt_files = sorted(list(model_dir.glob("best_fold*.pth")))
    if not ckpt_files:
        print(f"  No checkpoints found in {model_dir}")
        return None
        
    for ckpt_path in ckpt_files:
        try:
            # Use weights_only=False because metrics might be custom objects/dicts
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            metrics = checkpoint.get("metrics", {})
            
            # Normalizing metric keys (some might be lowercase, some uppercase)
            m = {k.lower(): v for k, v in metrics.items()}
            
            # If MSE isn't there, calculate it from RMSE
            if "mse" not in m and "rmse" in m:
                m["mse"] = m["rmse"]**2
                
            fold_results.append(m)
            print(f"  {ckpt_path.name}: MAE={m.get('mae', 'N/A'):.4f}, RMSE={m.get('rmse', 'N/A'):.4f}, R2={m.get('r2', 'N/A'):.4f}")
        except Exception as e:
            print(f"  Error reading {ckpt_path.name}: {e}")
            
    if not fold_results:
        return None
        
    # Calculate averages
    avg = {}
    for key in ["mae", "rmse", "mse", "r2"]:
        vals = [res[key] for res in fold_results if key in res and res[key] is not None]
        if vals:
            avg[key] = np.mean(vals)
            avg[f"{key}_std"] = np.std(vals)
            
    return avg

all_stats = {}
for m_name in MODELS:
    stats = extract_metrics(CKPT_ROOT / m_name)
    if stats:
        all_stats[m_name] = stats

print("\n" + "="*50)
print("FINAL CONSOLIDATED METRICS (5-FOLD AVG)")
print("="*50)
print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'MSE':<10} | {'R2':<8}")
print("-" * 65)
for name, s in all_stats.items():
    print(f"{name:<20} | {s['mae']:8.2f} | {s['rmse']:8.2f} | {s['mse']:10.2f} | {s['r2']:8.4f}")
