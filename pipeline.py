from pathlib import Path
from src.cleaning.pipeline import clean_dataset
from src.common.io import load_dataset, save_dataset
from src.generator.pipeline import generate_dataset
import random

random.seed(42)

dataset = load_dataset(Path("data/raw"))
print('=' * 50)

clean_dataset(dataset)
print('=' * 50)

save_dataset(dataset, Path("data/cleaned"))
print('=' * 50)

generate_dataset(Path("data/generated"), count=500)
print('=' * 50)