from src.cleaning.rules import remove_points_with_long_barheight, remove_points_with_error_bars_beyond_image
from src.common.models import ImageAnnotation
from typing import List

def _total_points(dataset: List[ImageAnnotation]) -> int:
  return sum(len(line.points) for ann in dataset for line in ann.lines)

def clean_dataset(dataset: List[ImageAnnotation]) -> List[ImageAnnotation]:
  print("Starting dataset cleaning...")
  len_before = _total_points(dataset)
  dataset = remove_points_with_long_barheight(dataset)
  dataset = remove_points_with_error_bars_beyond_image(dataset)
  len_after = _total_points(dataset)
  print(f"Total points removed: {len_before} - {len_after} = {len_before - len_after}")
  return dataset