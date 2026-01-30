import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from src.detection.dataset import prepare_patches, ErrorBarPatchDataset
from src.detection.model import ErrorBarRegressor

def evaluate():
    ROOT_DIR = Path(__file__).parent
    RAW_DATA_DIR = ROOT_DIR / "data" / "cleaned"
    RAW_PATCHES_DIR = ROOT_DIR / "data" / "cleaned_patches"
    
    MODEL_PATH = ROOT_DIR / "error_bar_model.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting Evaluation on Delineate Dataset")
    print(f"Device: {DEVICE}")
    if not RAW_DATA_DIR.exists():
        print(f"Error: Could not find {RAW_DATA_DIR}")
        print("Please ensure you have 'data/raw/images' and 'data/raw/labels'")
        return

    print(f"\n[1/3]")
    if RAW_PATCHES_DIR.exists():
        print(f"Found existing patches. Skipping patch preparation.")
    else:
        prepare_patches(RAW_DATA_DIR, RAW_PATCHES_DIR)

    print(f"\n[2/3]")
    dataset = ErrorBarPatchDataset(RAW_PATCHES_DIR / "metadata.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Found {len(dataset)} valid data points to test.")

    model = ErrorBarRegressor().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    model.eval()

    print(f"\n[3/3]\nRunning Inference...")
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
    
    abs_errors = np.abs(preds - targets)
    
    mae = np.mean(abs_errors)
    mae_top = np.mean(abs_errors[:, 0])
    mae_bot = np.mean(abs_errors[:, 1])

    within_2px = np.mean(abs_errors < 2.0) * 100
    within_5px = np.mean(abs_errors < 5.0) * 100
    within_10px = np.mean(abs_errors < 10.0) * 100

    print("\n" + "="*50)
    print("FINAL DELINEATE DATASET RESULTS")
    print("="*50)
    print(f"Total Test Points: {len(dataset)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} pixels")
    print(f"  - Top Bar MAE: {mae_top:.2f} px")
    print(f"  - Bot Bar MAE: {mae_bot:.2f} px")
    print("-" * 30)
    print("Precision Breakdown:")
    print(f"  - Perfect (< 2px error):   {within_2px:.1f}%")
    print(f"  - Great   (< 5px error):   {within_5px:.1f}%")
    print(f"  - Good    (< 10px error):  {within_10px:.1f}%")
    print("="*50)
    
    print(f"\n[4/4]\nGenerating Failure Analysis Report...")
    
    # 1. Create a detailed DataFrame
    report_df = dataset.metadata.copy()
    report_df['pred_top'] = preds[:, 0]
    report_df['pred_bot'] = preds[:, 1]
    report_df['true_top'] = targets[:, 0]
    report_df['true_bot'] = targets[:, 1]
    report_df['err_top'] = abs_errors[:, 0]
    report_df['err_bot'] = abs_errors[:, 1]
    report_df['max_error'] = np.max(abs_errors, axis=1)

    # 2. Save CSV
    report_path = ROOT_DIR / "evaluation_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"📝 Detailed CSV saved to: {report_path}")

    # 3. Visualize Top 20 Failures
    FAILURE_DIR = ROOT_DIR / "failure_analysis"
    if FAILURE_DIR.exists():
        shutil.rmtree(FAILURE_DIR)
    FAILURE_DIR.mkdir()

    # Get indices of the 20 worst errors
    top_failures = report_df.nlargest(20, 'max_error')
    
    print(f"📸 Saving top 20 worst failure plots to {FAILURE_DIR}...")
    
    for idx, row in top_failures.iterrows():
        # Retrieve the specific patch using the dataset
        # Note: dataset[idx] returns (image_tensor, target_tensor)
        img_tensor, _ = dataset[idx] 
        
        # Denormalize image for display: (img * 0.5) + 0.5
        img_np = img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1)

        plt.figure(figsize=(2, 10))
        plt.imshow(img_np)
        
        center_y = 400 # Patch height is 800, so center is 400
        
        # Draw True (Green)
        plt.axhline(center_y - row['true_top'], color='green', linewidth=2, label='True')
        plt.axhline(center_y + row['true_bot'], color='green', linewidth=2)
        
        # Draw Pred (Red)
        plt.axhline(center_y - row['pred_top'], color='red', linestyle='--', linewidth=2, label='Pred')
        plt.axhline(center_y + row['pred_bot'], color='red', linestyle='--', linewidth=2)
        
        plt.title(f"Err: {row['max_error']:.1f}px", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        safe_name = Path(row['filename']).stem
        plt.savefig(FAILURE_DIR / f"rank_{int(row['max_error'])}px_{safe_name}.png")
        plt.close()

    print("Done.")

if __name__ == "__main__":
    evaluate()