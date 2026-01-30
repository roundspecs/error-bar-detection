import os
import json
import uuid
import torch
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from src.config import PATCH_H, PATCH_W

class ErrorBarPatchDataset(Dataset):
    """PyTorch Dataset that loads pre-cropped patches."""
    def __init__(self, metadata_csv: Path, transform=None, augment=False):
        """
        Args:
            augment (bool): If True, applies random noise/blur (Use for Training).
                            If False, only normalizes (Use for Validation/Testing).
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.root_dir = metadata_csv.parent / "images"
        
        # Base transform
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if augment:
            self.transform = transforms.Compose([
                # 1. Blur
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
                ], p=0.3),
                
                # 2. Lighting/Color Noise
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                
                # 3. Standard formatting
                self.base_transform
            ])
        else:
            self.transform = self.base_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.root_dir / row['filename']
        image = Image.open(img_path).convert("RGB")
        
        targets = torch.tensor([
            float(row['top_dist']), 
            float(row['bottom_dist'])
        ], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets

def pad_and_crop(image: Image.Image, cx: int, cy: int, h: int, w: int) -> Image.Image:
    """
    Crops a patch of size (h, w) centered at (cx, cy).
    If the crop goes out of bounds, it pads with the edge color.
    """
    img_w, img_h = image.size
    
    # Calculate crop coordinates
    left = cx - w // 2
    top = cy - h // 2
    right = left + w
    bottom = top + h
    
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - img_w)
    pad_bottom = max(0, bottom - img_h)
    
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        # Pad with the border color (replicates the edge pixels)
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        image = ImageOps.expand(image, padding)
        
        left += pad_left
        top += pad_top
        right += pad_left
        bottom += pad_top

    return image.crop((left, top, right, bottom))

def prepare_patches(source_root: Path, dest_root: Path):
    """
    One-time preprocessing script.
    Reads synthetic images/labels -> Crops patches -> Saves to dest_root.
    """
    source_images = source_root / "images"
    source_labels = source_root / "labels"
    
    dest_images = dest_root / "images"
    dest_images.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    print(f"Processing dataset from {source_root}...")
    
    label_files = list(source_labels.glob("*.json"))
    
    for label_path in tqdm(label_files, desc="Cropping patches"):
        # Match label with image
        img_name = label_path.stem + ".png"
        img_path = source_images / img_name
        
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path) as img:
                with open(label_path, "r") as f:
                    data = json.load(f)
                
                for line in data:
                    for point in line.get("points", []):
                        if point.get('label') in ['xmin', 'xmax', 'ymin', 'ymax']:
                            continue
                        cx = int(point['x'])
                        cy = int(point['y'])
                        
                        patch_id = str(uuid.uuid4())
                        patch_filename = f"{patch_id}.png"
                        
                        patch = pad_and_crop(img, cx, cy, PATCH_H, PATCH_W)
                        patch.save(dest_images / patch_filename)
                        
                        metadata.append({
                            "filename": patch_filename,
                            "original_image": img_name,
                            "line_name": line['label']['lineName'],
                            "src_x": cx,
                            "src_y": cy,
                            "top_dist": point['topBarPixelDistance'],
                            "bottom_dist": point['bottomBarPixelDistance']
                        })
                        
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    df = pd.DataFrame(metadata)
    csv_path = dest_root / "metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} patches to {dest_root}")
    print(f"Metadata saved to {csv_path}")