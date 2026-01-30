import uuid
import random
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from config import (
    DPI_OPTIONS,
    GEN_ASYMMETRIC_PROB,
    GEN_LONG_FRACTION,
    GEN_MISSING_BAR_PROB,
    PLOT_TYPE_PROBS,
)


def generate_image():
    """Generates one image"""
    id = str(uuid.uuid4())
    w = random.randint(8, 15)
    h = random.randint(6, 12)
    dpi = random.choice(DPI_OPTIONS)
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    set_background(ax)
    set_spines(ax)

    long_mode = random.random() < GEN_LONG_FRACTION

    # Decide Log Scale (Common to line/bar)
    is_log_y = False
    if dice < PLOT_TYPE_PROBS["linegraph"] and random.random() > 0.6:
        is_log_y = True
        ax.set_yscale("log")

    dice = random.random()
    if dice < PLOT_TYPE_PROBS["linegraph"]:
        plot_data = generate_linegraph(ax, is_log_y, long_mode, h, dpi)
    elif dice < PLOT_TYPE_PROBS["linegraph"] + PLOT_TYPE_PROBS["barchart"]:
        plot_data = generate_barchart(ax)
    else:
        plot_data = generate_boxplot(ax)

    ax.set_title(f"Generated Plot {uuid.uuid4().hex[:6]}")
    if random.random() > 0.2:
        locs = ["best", "upper right", "upper left", "lower right"]
        ax.legend(loc=random.choice(locs))
        
    return fig, plot_data


def set_background(ax):
    if random.random() > 0.7:
        ax.set_facecolor(random.choice(["#f0f0f0", "#e0e0e0", "#ebebeb"]))  # Light gray
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    else:
        ax.set_facecolor("white")
        if random.random() > 0.5:
            ax.grid(True, linestyle=":", alpha=0.4, color="gray")


def set_spines(ax):
    if random.random() > 0.4:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if random.random() > 0.8:  # Occasional floating axes
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def generate_linegraph(ax, is_log_y, long_mode, h, dpi):
    num_series = random.randint(1, 5)
    all_lines_data = []

    is_grayscale = random.random() < 0.20
    colors = (
        ["black", "gray"]
        if is_grayscale
        else ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    )

    for i in range(num_series):
        num_points = random.randint(4, 12)
        x, y = generate_data_points(num_points, is_line=True, is_log_y=is_log_y)
        top, bot = calculate_error_bars(y, is_log_y, h, dpi, long_mode)

        color = random.choice(colors)
        marker = random.choice(["o", "s", "^", "v", "D"])
        linestyle = random.choice(["-", "--", "-."]) if not is_grayscale else "-"

        ax.errorbar(
            x,
            y,
            yerr=[bot, top],
            label=f"Group_{i}",
            fmt=marker,
            color=color,
            capsize=random.randint(2, 6),
            linestyle=linestyle,
        )

        all_lines_data.append(
            {
                "label": f"Group_{i}",
                "_raw_data": list(zip(x, y, top, bot)),  # x, y, top_err, bot_err
            }
        )

    return all_lines_data


def generate_barchart(ax):
    pass


def generate_boxplot(ax):
    pass


def generate_data_points(num_points, is_line=True, is_log_y=False):
    """Generates X and Y values."""
    if is_line and random.random() > 0.3:
        pool = range(0, 30)
        x_points = np.sort(random.sample(pool, num_points))
    else:
        x_points = np.arange(num_points)

    y_points = np.random.uniform(10, 100, size=len(x_points))
    if is_log_y:
        y_points = y_points + 10

    return x_points, y_points


def calculate_error_bars(y_points, is_log_y, h, dpi, long_mode):
    """Calculates top and bottom error values (in data units)."""
    data_span = np.max(y_points) - np.min(y_points)
    if data_span == 0:
        data_span = np.mean(y_points) * 0.1
    plot_h_pixels = h * dpi * 0.8
    units_per_pixel = data_span / plot_h_pixels

    top_err, bot_err = [], []

    for val_y in y_points:
        if random.random() < GEN_MISSING_BAR_PROB:
            t, b = 0.0, 0.0
        else:
            is_asymmetric = random.random() < GEN_ASYMMETRIC_PROB
            px_top = get_random_pixel_length(long_mode, h, dpi)

            if is_asymmetric:
                roll = random.random()
                if roll < 0.10:
                    px_bot = 0.0
                elif roll < 0.20:
                    px_bot = get_random_pixel_length(long_mode, h, dpi)
                    px_top = 0.0
                else:
                    px_bot = get_random_pixel_length(long_mode, h, dpi)
            else:
                px_bot = px_top

            # Convert pixel length to data units
            if is_log_y:
                rel_err_top = px_top / plot_h_pixels
                rel_err_bot = px_bot / plot_h_pixels
                t = val_y * (10**rel_err_top - 1) if px_top > 0 else 0
                b = val_y * (1 - 10 ** (-rel_err_bot)) if px_bot > 0 else 0
            else:
                t = px_top * units_per_pixel
                b = px_bot * units_per_pixel

            if t > 0:
                t = max(t, val_y * 0.001)
            if b > 0:
                b = max(b, val_y * 0.001)

        top_err.append(t)
        bot_err.append(b)

    return top_err, bot_err


def get_random_pixel_length(long_mode, h, dpi):
    """Helper to determine error bar length based on generation mode."""
    dice_roll = random.random()
    if long_mode:
        if dice_roll < 0.30:
            return abs(np.random.normal(loc=60, scale=20))
        elif dice_roll < 0.75:
            return abs(np.random.uniform(50, max(80, h * dpi * 0.18)))
        else:
            return abs(np.random.uniform(80, h * dpi * 0.25))
    else:
        if dice_roll < 0.60:
            return abs(np.random.gamma(shape=1.0, scale=12.0))
        elif dice_roll < 0.90:
            return abs(np.random.normal(loc=35, scale=10))
        else:
            return abs(np.random.uniform(60, h * dpi * 0.2))
