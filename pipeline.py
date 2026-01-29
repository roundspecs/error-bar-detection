from pathlib import Path
from src.cleaning.clean import clean_dataset
from src.common.io import load_dataset, save_dataset

dataset = load_dataset(Path("data/raw"))
print('=' * 50)

clean_dataset(dataset)
print('=' * 50)

save_dataset(dataset, Path("data/cleaned"))
print('=' * 50)

