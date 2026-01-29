from generator import generate_image
import random

def run_generation_pipeline(output_dir: str, count: int):
  """Runs the image generation pipeline."""
  random.seed(42)
  for _ in range(count):
    generate_image(output_dir)