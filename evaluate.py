import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.detection.dataset import prepare_patches, ErrorBarPatchDataset
from src.detection.model import ErrorBarRegressor

def evaluate():
    ROOT_DIR = Path(__file__).parent
    RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
    RAW_PATCHES_DIR = ROOT_DIR / "data" / "raw_patches"
    
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
    
if __name__ == "__main__":
    evaluate()