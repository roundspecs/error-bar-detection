import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import time

from src.generator.pipeline import generate_dataset
from src.detection.dataset import prepare_patches, ErrorBarPatchDataset
from src.detection.model import ErrorBarRegressor
from config import NUM_GENERATED_IMAGES, BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_SPLIT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Starting Pipeline on device: {DEVICE}")
    
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
    if generated_image:
        prepare_patches(DATA_RAW, DATA_PATCHES)
    else:
        print(f"Found {len(existing_images)} existing images. Skipping patch preparation.")
    
    print(f"\n[3/4]")
    full_dataset = ErrorBarPatchDataset(DATA_PATCHES / "metadata.csv")
    
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    print(f"Training on {len(train_data)} patches. Validating on {len(val_data)} patches.")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = ErrorBarRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ New best model saved!")

    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.1f}s.")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()