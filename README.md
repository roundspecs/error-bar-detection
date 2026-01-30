# Error Bar Detection

Developed for the Delineate AI/ML Intern Assignment: https://github.com/delineate-pro/error-bar-detection-assignment

## Overview

This repository implements an automated error bar detection system for scientific plots using deep learning. The project includes tools for data generation, cleaning, model training, evaluation, and inference.

The system uses a CNN-based regression model to predict error bar positions (top and bottom distances) from data points in scientific plots.

## Project Structure

```
error-bar-detection/
├── src/                        # Source code modules
│   ├── cleaning/              # Data cleaning and validation
│   │   ├── cleaner.py         # Main cleaning orchestrator
│   │   └── rules.py           # Cleaning rules and logic
│   ├── common/                # Shared utilities
│   │   ├── io.py              # File I/O operations
│   │   ├── models.py          # Data models (Pydantic)
│   │   └── utils.py           # Helper functions
│   ├── detection/             # Model and dataset for detection
│   │   ├── dataset.py         # PyTorch dataset and patch extraction
│   │   └── model.py           # CNN regression model
│   ├── generator/             # Synthetic data generation
│   │   ├── generate.py        # Dataset generation logic
│   │   └── pipeline.py        # Generation pipeline
│   └── config.py              # Global configuration parameters
├── scripts/                    # Executable scripts (see below)
├── data/                       # Data directory (created during execution)
│   ├── raw/                   # Raw input data
│   ├── cleaned/               # Cleaned data
│   ├── generated/             # Generated synthetic data
│   ├── cleaned_patches/       # Patches from cleaned data
│   └── generated_patches/     # Patches from generated data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/delineate-pro/error-bar-detection-assignment
cd error-bar-detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Scripts Usage

All scripts are located in the `scripts/` directory and should be run from the project root.

### 1. Generate Synthetic Data

**Script:** `scripts/generate.py`

Generates synthetic scientific plots with known error bar positions for training.

```bash
python -m scripts.generate
```

**What it does:**
- Creates synthetic line graphs, bar charts, and box plots
- Generates plots with varying DPI, colors, and error bar configurations
- Saves images to `data/generated/images/`
- Saves metadata (ground truth) to `data/generated/metadata.json`

**Configuration:**
Edit `src/config.py` to customize:
- `NUM_GENERATED_IMAGES`: Number of images to generate (default: 30)
- `PLOT_TYPE_PROBS`: Distribution of plot types
- `DPI_OPTIONS`: Resolution options
- `GEN_MISSING_BAR_PROB`: Probability of missing error bars
- `GEN_ASYMMETRIC_PROB`: Probability of asymmetric error bars

### 2. Clean Raw Data

**Script:** `scripts/clean.py`

Cleans and validates raw annotated data from human labelers.

```bash
python -m scripts.clean
```

**What it does:**
- Loads data from `data/raw/`
- Applies cleaning rules (removes outliers, validates annotations, fixes phantom bars)
- Saves cleaned data to `data/cleaned/`

**Input format:** The raw data should follow the structure defined in `src/common/models.py` with images in `data/raw/images/` and metadata in `data/raw/metadata.json`.

**Cleaning rules (from `src/cleaning/rules.py`):**
- Removes error bars exceeding maximum height
- Filters phantom/noise error bars
- Validates data point positions

### 3. Train the Model

**Script:** `scripts/train.py`

Trains the error bar detection model on generated data.

```bash
python -m scripts.train
```

**What it does:**
1. Generates synthetic data (if not already present)
2. Extracts patches around data points
3. Splits data into training/validation sets (80/20)
4. Trains a CNN regression model
5. Saves the best model to `error_bar_model.pth`

**Training configuration (in `src/config.py`):**
- `BATCH_SIZE`: 64
- `LEARNING_RATE`: 0.001
- `EPOCHS`: 2 (increase for production)
- `TRAIN_SPLIT`: 0.8

**Output:**
- Model checkpoint: `error_bar_model.pth`
- Training logs showing loss per epoch

**Device support:** Automatically uses CUDA if available, otherwise CPU.

### 4. Evaluate the Model

**Script:** `scripts/evaluate.py`

Evaluates model performance on the cleaned dataset.

**Prerequisite:** You must run `scripts/clean.py` first to generate the cleaned dataset in `data/cleaned/`.

```bash
python -m scripts.evaluate
```

**What it does:**
1. Loads the trained model from `error_bar_model.pth`
2. Runs inference on cleaned test data
3. Calculates metrics:
   - Mean Absolute Error (MAE) for top and bottom error bars
   - Detailed statistics (min, max, percentiles)
   - Ghost bar analysis (errors on missing vs. present bars)
4. Generates failure analysis visualizations
5. Saves worst-performing predictions to `failure_analysis/`

**Output:**
```
=== FINAL RESULTS ===
Mean Absolute Error (MAE): X.XX px
  - Top Bar MAE: X.XX px
  - Bot Bar MAE: X.XX px

=== Ghost Bar Analysis ===
Mean Error on REAL bars:    X.XX px
Mean Error on MISSING bars: X.XX px
```

**Failure analysis images:** Located in `failure_analysis/` directory, named by error magnitude.

### 5. Run Inference (Predict)

**Script:** `scripts/predict.py`

Runs error bar detection on a single plot image.

```bash
python -m scripts.predict \
  --image path/to/plot.png \
  --input path/to/input.json \
  --output path/to/output.json \
  --model error_bar_model.pth
```

**Arguments:**
- `--image` (required): Path to the plot image (PNG format)
- `--input` (required): Path to input JSON with data point coordinates
- `--output` (optional): Path to save predictions (default: `output.json`)
- `--model` (optional): Path to trained model (default: `error_bar_model.pth`)

**Input JSON format:**
```json
{
  "image_file": "sample_plot.png",
  "data_points": [
    {
      "lineName": "Line 1",
      "points": [
        {"x": 100, "y": 200},
        {"x": 150, "y": 180}
      ]
    }
  ]
}
```

**Output JSON format:**
```json
{
  "image_file": "sample_plot.png",
  "error_bars": [
    {
      "lineName": "Line 1",
      "points": [
        {
          "data_point": {"x": 100, "y": 200},
          "upper_error_bar": {"x": 100, "y": 170},
          "lower_error_bar": {"x": 100, "y": 230}
        }
      ]
    }
  ]
}
```

**Example:**
```bash
python -m scripts.predict \
  --image sample_plot.png \
  --input sample_input.json \
  --output sample_output_predicted.json
```

## Typical Workflow

### For Training a New Model:

1. Generate synthetic training data:
```bash
python -m scripts.generate
```

2. Train the model:
```bash
python -m scripts.train
```

3. Evaluate on test data (requires cleaned data):
```bash
python -m scripts.clean  # First, clean the raw data
python -m scripts.evaluate
```

### For Using an Existing Model:

1. Run inference on new plots:
```bash
python -m scripts.predict --image your_plot.png --input your_input.json
```

### For Working with Labeled Data:

1. Place raw data in `data/raw/`
2. Clean the data:
```bash
python -m scripts.clean
```

3. Use cleaned data for evaluation or additional training

## Configuration

All model and data parameters are centralized in `src/config.py`. Key settings include:

**Cleaning:**
- `MAX_BARHEIGHT`: Maximum allowed error bar height (430 px)
- `MINIMUM_PHANTOM_DISTANCE`: Minimum distance to avoid phantom bars (15 px)

**Generation:**
- `NUM_GENERATED_IMAGES`: Number of synthetic images (30)
- `PLOT_TYPE_PROBS`: Distribution of plot types
- `GEN_MISSING_BAR_PROB`: Probability of missing bars (0.45)
- `GEN_ASYMMETRIC_PROB`: Probability of asymmetric bars (0.65)

**Detection:**
- `PATCH_H`: Patch height (800 px)
- `PATCH_W`: Patch width (64 px)

**Training:**
- `BATCH_SIZE`: Training batch size (64)
- `LEARNING_RATE`: Optimizer learning rate (0.001)
- `EPOCHS`: Number of training epochs (2)
- `TRAIN_SPLIT`: Train/val split ratio (0.8)

## Model Architecture

The error bar detection model (`src/detection/model.py`) uses a CNN-based architecture:

- Input: 800x64 RGB patches centered on data points
- Output: 2 values (top_distance, bottom_distance)
- Loss: L1 Loss (MAE)

## Dependencies

Core dependencies (from `requirements.txt`):
- `matplotlib`: Plot generation and visualization
- `pandas`: Data manipulation
- `torch`: Deep learning framework
- `torchvision`: Pre-trained models and transforms
- `tqdm`: Progress bars
- `pydantic`: Data validation

## Troubleshooting

**CUDA/GPU Issues:**
The scripts automatically detect and use CUDA if available. If you encounter GPU issues, the system will fall back to CPU.

**Missing Data Directories:**
Scripts automatically create required directories. Ensure you have write permissions in the project directory.

**Model Not Found:**
Run `python scripts/train.py` to generate `error_bar_model.pth` before evaluation or prediction.

**Import Errors:**
Ensure you're running scripts from the project root directory, not from within the `scripts/` folder.

## License

This project is developed for the Delineate AI/ML Intern Assignment.
