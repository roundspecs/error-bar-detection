from pathlib import Path
import json
from src.common.models import ImageAnnotation, DatasetLine, DatasetPoint
import shutil
from typing import List

def load_dataset(dataset_root: Path):
  """Loads the dataset from the given root directory."""
  print(f"Loading dataset from {dataset_root}...")
  labels_dir = dataset_root / "labels"
  images_dir = dataset_root / "images"

  json_files = sorted(labels_dir.glob("*.json"))

  dataset = []

  for json_file in json_files:
    image_file = images_dir / f"{json_file.stem}.png"
    with open(json_file, "r") as f:
      raw_data = json.load(f)
      lines = [DatasetLine(**line_data) for line_data in raw_data]
      annotation = ImageAnnotation(
          id=json_file.stem,
          image_path=image_file,
          json_path=json_file,
          lines=lines
      )
      dataset.append(annotation)

  return dataset


def save_dataset(dataset: List[ImageAnnotation], output_root: Path):
    """
    Saves a list of ImageAnnotation objects to disk in the standard dataset format.
    """
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(dataset)} items to {output_root}...")

    for image in dataset:
        dest_image_path = images_dir / image.image_path.name
        
        if image.image_path.resolve() != dest_image_path.resolve():
          try:
              shutil.copy2(image.image_path, dest_image_path)
          except FileNotFoundError:
              print(f"Warning: Source image missing for {image.image_id}. Skipping copy.")
              return

        json_data = [line.model_dump(exclude_none=True) for line in image.lines]
        
        dest_json_path = labels_dir / image.json_path.name
        with open(dest_json_path, "w") as f:
            json.dump(json_data, f, indent=2)