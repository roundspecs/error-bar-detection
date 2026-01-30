import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import sys

# Ensure src can be imported if running from scripts/ directory
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from src.detection.dataset import prepare_patches, ErrorBarPatchDataset
from src.detection.model import ErrorBarRegressor

def evaluate():
    # 1. Fix Path Logic
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data" / "cleaned"
    PATCHES_DIR = ROOT_DIR / "data" / "cleaned_patches"
    MODEL_PATH = ROOT_DIR / "error_bar_model.pth" # Or check output/models/ if you saved it there
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting Evaluation")
    print(f"Device: {DEVICE}")
    
    if not DATA_DIR.exists():
        print(f"❌ Error: Could not find dataset at {DATA_DIR}")
        return

    # [1/3] Prepare Patches
    print(f"\n[1/3] Checking Patches...")
    if PATCHES_DIR.exists():
        print(f"Found existing patches at {PATCHES_DIR.name}")
    else:
        print("Generating patches from images...")
        prepare_patches(DATA_DIR, PATCHES_DIR)

    # [2/3] Load Data & Model
    print(f"\n[2/3] Loading Model & Data...")
    dataset = ErrorBarPatchDataset(PATCHES_DIR / "metadata.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Test Set Size: {len(dataset)} points")

    model = ErrorBarRegressor().to(DEVICE)
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # [3/3] Inference
    print(f"\n[3/3] Running Inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    
    # --- STATISTICS GENERATION ---
    df = pd.DataFrame({
        'true_top': targets[:, 0],
        'true_bot': targets[:, 1],
        'pred_top': preds[:, 0],
        'pred_bot': preds[:, 1]
    })

    # Calculate absolute errors
    df['err_top'] = np.abs(df['true_top'] - df['pred_top'])
    df['err_bot'] = np.abs(df['true_bot'] - df['pred_bot'])
    df['max_error'] = df[['err_top', 'err_bot']].max(axis=1)

    # --- EXPLICIT MAE CALCULATION ---
    mae_top = df['err_top'].mean()
    mae_bot = df['err_bot'].mean()
    # Global MAE is the mean of ALL error components (Top + Bot) combined
    mae_global = (df['err_top'].sum() + df['err_bot'].sum()) / (2 * len(df))

    print("\n" + "="*50)
    print("=== 🏆 FINAL RESULTS ===")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae_global:.2f} px")
    print(f"  - Top Bar MAE: {mae_top:.2f} px")
    print(f"  - Bot Bar MAE: {mae_bot:.2f} px")
    print("-" * 50)

    print("\n=== 📊 Detailed Statistics (Pixels) ===")
    print(df[['err_top', 'err_bot', 'max_error']].describe().round(2))

    # --- GHOST BAR ANALYSIS ---
    flat_targets = np.concatenate([df['true_top'].values, df['true_bot'].values])
    flat_errors = np.concatenate([df['err_top'].values, df['err_bot'].values])

    mask_missing = (flat_targets == 0)
    mask_real = (flat_targets > 0)

    avg_real_err = np.mean(flat_errors[mask_real]) if np.any(mask_real) else 0.0
    avg_miss_err = np.mean(flat_errors[mask_missing]) if np.any(mask_missing) else 0.0

    print("\n=== 👻 Ghost Bar Analysis ===")
    print(f"Mean Error on REAL bars:    {avg_real_err:.2f} px")
    print(f"Mean Error on MISSING bars: {avg_miss_err:.2f} px")
    print("="*50)

    # --- FAILURE VISUALIZATION ---
    print(f"\n[4/4] Generating Failure Analysis Images...")
    FAILURE_DIR = ROOT_DIR / "failure_analysis"
    if FAILURE_DIR.exists():
        shutil.rmtree(FAILURE_DIR)
    FAILURE_DIR.mkdir()

    top_failures = df.nlargest(20, 'max_error')
    
    print(f"📸 Saving top 20 worst failure plots to {FAILURE_DIR.name}/...")
    
    for idx, row in top_failures.iterrows():
        img_tensor, _ = dataset[idx] 
        img_np = img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1)

        plt.figure(figsize=(2, 10))
        plt.imshow(img_np)
        
        center_y = 400 
        
        plt.axhline(center_y - row['true_top'], color='green', linewidth=2, label='True')
        plt.axhline(center_y + row['true_bot'], color='green', linewidth=2)
        plt.axhline(center_y - row['pred_top'], color='red', linestyle='--', linewidth=2, label='Pred')
        plt.axhline(center_y + row['pred_bot'], color='red', linestyle='--', linewidth=2)
        
        plt.title(f"Err: {row['max_error']:.1f}px", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        
        safe_name = dataset.metadata.iloc[idx]['filename']
        safe_name = Path(safe_name).stem
        plt.savefig(FAILURE_DIR / f"rank_{int(row['max_error'])}px_{safe_name}.png")
        plt.close()

    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    evaluate()