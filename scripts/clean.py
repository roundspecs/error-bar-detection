import random
from src.cleaning.cleaner import clean_dataset
from src.common.io import load_dataset, save_dataset
from pathlib import Path

random.seed(42)

dataset = load_dataset(Path("data/raw"))
print('=' * 50)

clean_dataset(dataset)
print('=' * 50)

save_dataset(dataset, Path("data/cleaned"))
print('=' * 50)