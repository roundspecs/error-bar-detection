from pathlib import Path
from src.cleaning.clean import clean_dataset
from src.common.loader import load_dataset

dataset = load_dataset(Path("data/raw"))
print('=' * 50)

clean_dataset(dataset)
print('=' * 50)

