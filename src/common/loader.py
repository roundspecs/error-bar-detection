from pathlib import Path
import json
from src.common.models import ImageAnnotation, DatasetLine, DatasetPoint

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

