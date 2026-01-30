import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset  # Changed random_split to Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import time
import random
import numpy as np
from src.generator.pipeline import generate_dataset
from src.detection.dataset import prepare_patches, ErrorBarPatchDataset
from src.detection.model import ErrorBarRegressor
from config import NUM_GENERATED_IMAGES, BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_SPLIT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Starting Pipeline on device: {DEVICE}")

    # Make it reproducible
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    
    ROOT_DIR = Path(__file__).parent
    DATA_RAW = ROOT_DIR / "data" / "generated"
    DATA_PATCHES = ROOT_DIR / "data" / "patches"
    MODEL_SAVE_PATH = ROOT_DIR / "error_bar_model.pth"

    print(f"\n[1/4]")
    generated_image = False
    existing_images = list((DATA_RAW / "images").glob("*.png")) if DATA_RAW.exists() else []
    
    if len(existing_images) < NUM_GENERATED_IMAGES:
        generate_dataset(DATA_RAW, count=NUM_GENERATED_IMAGES)
        generated_image = True
    else:
        print(f"Found {len(existing_images)} existing images. Skipping generation.")

    print(f"\n[2/4]")
    if generated_image or not (DATA_PATCHES / "metadata.csv").exists():
        prepare_patches(DATA_RAW, DATA_PATCHES)
    else:
        print(f"Patches already exist. Skipping patch preparation.")
    
    print(f"\n[3/4]")
    csv_path = DATA_PATCHES / "metadata.csv"

    train_ds_base = ErrorBarPatchDataset(csv_path, augment=True)
    val_ds_base = ErrorBarPatchDataset(csv_path, augment=False)
    
    dataset_len = len(train_ds_base)
    indices = list(range(dataset_len))
    split_idx = int(TRAIN_SPLIT * dataset_len)
    
    np.random.shuffle(indices)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = Subset(train_ds_base, train_indices)
    val_data = Subset(val_ds_base, val_indices)
    
    print(f"Training on {len(train_data)} patches. Validating on {len(val_data)} patches.")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = ErrorBarRegressor().to(DEVICE)
    criterion = nn.L1Loss() # Changed to L1Loss for better outlier handling
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    print(f"\n[4/4]\nStarting Training for {EPOCHS} epochs...")
    start_time = time.time()
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, targets in pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.2f}"})
            
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Results - Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ New best model saved!")

    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.1f}s.")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()