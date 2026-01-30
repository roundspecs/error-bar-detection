import json
import uuid
from pathlib import Path
import matplotlib.pyplot as plt
from src.generator.generator import generate_image
from src.common.utils import delete_dir
from tqdm import tqdm

def generate_dataset(output_dir: Path, count: int):
    """Runs the image generation pipeline."""
    print(f"Generating {count} samples into {output_dir}...")

    if output_dir.exists():
        delete_dir(output_dir)

    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(count)):
        try:
            fig, data = generate_image()

            file_id = str(uuid.uuid4())
            fig.savefig(img_dir / f"{file_id}.png")
            plt.close(fig)  # Important: Close the figure to free memory

            with open(lbl_dir / f"{file_id}.json", "w") as f:
                json.dump(data, f, indent=2)


        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            plt.close("all")
