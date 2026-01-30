from pathlib import Path
from src.generator.generate import generate_dataset
import random

random.seed(42)

generate_dataset(Path("data/generated"), count=3000)
print('=' * 50)