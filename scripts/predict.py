import argparse
import json
import torch
import sys
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms

from src.detection.model import ErrorBarRegressor
from src.config import PATCH_H, PATCH_W

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = Path(__file__).resolve().parent.parent

def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = ErrorBarRegressor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def extract_patch(image, x, y, h=PATCH_H, w=PATCH_W):
    """
    Extracts a patch centered at (x, y). 
    Pads with white if the patch goes out of bounds.
    """
    img_w, img_h = image.size
    
    # Calculate crop coordinates
    left = x - (w // 2)
    upper = y - (h // 2)
    right = left + w
    lower = upper + h
    
    pad_left = max(0, -left)
    pad_top = max(0, -upper)
    pad_right = max(0, right - img_w)
    pad_bottom = max(0, lower - img_h)
    
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        # Add white border
        image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill='white')
        # Adjust crop coordinates because image origin shifted by (pad_left, pad_top)
        left += pad_left
        upper += pad_top
        right += pad_left
        lower += pad_top

    patch = image.crop((left, upper, right, lower))
    return patch

def predict(image_path, input_json_path, output_json_path, model_path):
    print(f"--- Error Bar Detection Inference ---")
    print(f"Image: {image_path.name}")
    print(f"Device: {DEVICE}")

    model = load_model(model_path)
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    output_data = {
        "image_file": input_data.get("image_file", image_path.name),
        "error_bars": []
    }

    total_points = sum(len(line['points']) for line in input_data['data_points'])
    print(f"Processing {total_points} points...")

    with torch.no_grad():
        for line in input_data['data_points']:
            line_result = {
                "lineName": line['lineName'],
                "points": []
            }
            
            for pt in line['points']:
                px_x, px_y = pt['x'], pt['y']
                
                patch = extract_patch(image, int(px_x), int(px_y))
                input_tensor = transform(patch).unsqueeze(0).to(DEVICE)
                
                dists = model(input_tensor).cpu().numpy()[0]
                
                top_dist = max(0.0, float(dists[0]))
                bot_dist = max(0.0, float(dists[1]))
                
                upper_y = px_y - top_dist
                lower_y = px_y + bot_dist
                
                line_result["points"].append({
                    "data_point": {"x": px_x, "y": px_y},
                    "upper_error_bar": {"x": px_x, "y": upper_y},
                    "lower_error_bar": {"x": px_x, "y": lower_y}
                })
            
            output_data["error_bars"].append(line_result)

    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Prediction saved to: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict error bars for a single plot.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input PNG image")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSON (data points)")
    parser.add_argument("--output", type=Path, default="output.json", help="Path to save output JSON")
    parser.add_argument("--model", type=Path, default=ROOT_DIR / "error_bar_model.pth", help="Path to trained model")
    
    args = parser.parse_args()
    
    predict(args.image, args.input, args.output, args.model)