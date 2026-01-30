# Cleaning
MAX_BARHEIGHT = 430.0
MINIMUM_PHANTOM_DISTANCE = 15

# Generation
PLOT_TYPE_PROBS = {
    "linegraph": 0.88,
    "barchart": 0.10,
    "boxplot": 0.02,
}
DPI_OPTIONS = [72, 100, 150, 200]
GEN_MISSING_BAR_PROB = 0.20
GEN_ASYMMETRIC_PROB = 0.65
REALISTIC_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", # Tableau/Publication
    "#000000", "#333333",                                  # Standard Black/Dark Gray
    "#4472C4", "#ED7D31", "#A5A5A5",                       # Excel Default
    "#EE6677", "#228833", "#CCBB44", "#66CCEE"             # Prism-like brights
]
# Long Bar / Sanity Settings
GEN_LONG_FRACTION = 0.20          # Fraction of images biased toward long bars
GEN_LONG_THRESHOLD_PX = 40
MIN_PIXEL_NONZERO = 2             # Treat smaller distances as zero
GEN_MAX_LABEL_PX_FACTOR = 1.5     # Clamp labels to factor * height

# Detection
PATCH_H = 800  # 800px height covers ±400px from center, which is safe.
PATCH_W = 48   # Error bar caps are usually 10-20px wide, so 32px is safe.

# Training
NUM_GENERATED_IMAGES = 3000 
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
TRAIN_SPLIT = 0.8