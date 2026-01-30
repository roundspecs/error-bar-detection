# Cleaning
MAX_BARHEIGHT = 430.0
MINIMUM_PHANTOM_DISTANCE = 15

# Generation
PLOT_TYPE_PROBS = {
    "linegraph": 0.65,
    "barchart": 0.30,
    "boxplot": 0.05,
}
DPI_OPTIONS = [72, 100, 150, 200]
GEN_MISSING_BAR_PROB = 0.20
GEN_ASYMMETRIC_PROB = 0.65

# Long Bar / Sanity Settings
GEN_LONG_FRACTION = 0.20          # Fraction of images biased toward long bars
GEN_LONG_THRESHOLD_PX = 40
MIN_PIXEL_NONZERO = 2             # Treat smaller distances as zero
GEN_MAX_LABEL_PX_FACTOR = 1.5     # Clamp labels to factor * height