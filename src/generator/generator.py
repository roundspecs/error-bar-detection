import uuid
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cbook
from config import (
    DPI_OPTIONS,
    PLOT_TYPE_PROBS,
    GEN_MISSING_BAR_PROB,
    GEN_ASYMMETRIC_PROB,
    GEN_LONG_FRACTION,
    MIN_PIXEL_NONZERO,
    GEN_MAX_LABEL_PX_FACTOR,
)

def generate_image():
    """Generates one image and its corresponding label data."""
    w = random.randint(8, 15)
    h = random.randint(6, 12)
    dpi = random.choice(DPI_OPTIONS)
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    long_mode = random.random() < GEN_LONG_FRACTION
    
    set_background(ax)
    set_spines(ax)

    dice = random.random()
    plot_data = []

    is_log_y = False
    if dice < PLOT_TYPE_PROBS["linegraph"] and random.random() > 0.6:
        is_log_y = True
        ax.set_yscale("log")

    if dice < PLOT_TYPE_PROBS["linegraph"]:
        plot_data = generate_linegraph(ax, is_log_y, long_mode, h, dpi)
    elif dice < PLOT_TYPE_PROBS["linegraph"] + PLOT_TYPE_PROBS["barchart"]:
        plot_data = generate_barchart(ax, is_log_y, long_mode, h, dpi)
    else:
        plot_data = generate_boxplot(ax, is_log_y, long_mode, h, dpi)

    ax.set_title(f"Generated Plot {uuid.uuid4().hex[:6]}")
    
    handles, labels = ax.get_legend_handles_labels()
    if handles and random.random() > 0.2:
        locs = ["best", "upper right", "upper left", "lower right"]
        ax.legend(loc=random.choice(locs))

    add_occlusions(ax)

    fig.canvas.draw()
    final_json = calculate_labels(fig, ax, plot_data, h, is_log_y)

    return fig, final_json

def set_background(ax):
    if random.random() > 0.7:
        ax.set_facecolor(random.choice(["#f0f0f0", "#e0e0e0", "#ebebeb"]))
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
    if random.random() > 0.8:
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

def add_occlusions(ax):
    if random.random() < 0.18:
        rect_w = random.uniform(0.08, 0.25)
        rect_h = random.uniform(0.06, 0.18)
        rect_x = random.uniform(0.1, 0.7)
        rect_y = random.uniform(0.1, 0.7)
        oc = mpatches.Rectangle(
            (rect_x, rect_y), rect_w, rect_h,
            transform=ax.transAxes,
            facecolor=random.choice(["white", "#f8f8f8", "#f0f0f0"]),
            alpha=random.uniform(0.6, 0.95),
            zorder=5,
        )
        ax.add_patch(oc)

def get_random_pixel_length(long_mode, h, dpi):
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

def generate_data_points(num_points, is_line=True, is_log_y=False):
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
    data_span = np.max(y_points) - np.min(y_points)
    if data_span == 0: data_span = np.mean(y_points) * 0.1
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
                if roll < 0.10: px_bot = 0.0
                elif roll < 0.20: 
                    px_bot = get_random_pixel_length(long_mode, h, dpi)
                    px_top = 0.0
                else: px_bot = get_random_pixel_length(long_mode, h, dpi)
            else:
                px_bot = px_top

            if is_log_y:
                rel_err_top = px_top / plot_h_pixels
                rel_err_bot = px_bot / plot_h_pixels
                t = val_y * (10**rel_err_top - 1) if px_top > 0 else 0
                b = val_y * (1 - 10 ** (-rel_err_bot)) if px_bot > 0 else 0
            else:
                t = px_top * units_per_pixel
                b = px_bot * units_per_pixel

            if t > 0: t = max(t, val_y * 0.001)
            if b > 0: b = max(b, val_y * 0.001)

        top_err.append(t)
        bot_err.append(b)
    
    return top_err, bot_err

def generate_linegraph(ax, is_log_y, long_mode, h, dpi):
    num_series = random.randint(1, 5)
    all_lines_data = []
    
    is_grayscale = random.random() < 0.20
    colors = ["black", "gray"] if is_grayscale else ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    for i in range(num_series):
        num_points = random.randint(4, 12)
        x, y = generate_data_points(num_points, is_line=True, is_log_y=is_log_y)
        top, bot = calculate_error_bars(y, is_log_y, h, dpi, long_mode)
        
        color = random.choice(colors)
        marker = random.choice(["o", "s", "^", "v", "D"])
        linestyle = random.choice(["-", "--", "-."]) if not is_grayscale else "-"
        
        ax.errorbar(
            x, y, yerr=[bot, top], label=f"Group_{i}",
            fmt=marker, color=color, capsize=random.randint(2, 6),
            linestyle=linestyle
        )
        
        # Save Raw Data for Label Calculation
        all_lines_data.append({
            "label": {"lineName": f"Group_{i}"},
            "_raw_data": list(zip(x, y, top, bot))
        })
        
    return all_lines_data

def generate_barchart(ax, is_log_y, long_mode, h, dpi):
    num_series = random.randint(1, 2)
    all_lines_data = []
    
    is_grayscale = random.random() < 0.20
    colors = ["black", "gray"] if is_grayscale else ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    num_points = random.randint(4, 12)
    x_base = np.arange(num_points)
    width = 0.8 / num_series

    for i in range(num_series):
        _, y = generate_data_points(num_points, is_line=False, is_log_y=is_log_y)
        top, bot = calculate_error_bars(y, is_log_y, h, dpi, long_mode)
        
        x_pos = x_base + (i * width) - (0.4 if num_series > 1 else 0)
        color = random.choice(colors)
        
        face = "white"
        edge = color
        hatch = random.choice(["/", "\\", "x", "."]) if random.random() > 0.5 else None
        
        ax.bar(
            x_pos, y, width=width, label=f"Group_{i}",
            color=face, edgecolor=edge, hatch=hatch,
            yerr=[bot, top], ecolor="black", capsize=random.randint(2, 6)
        )

        all_lines_data.append({
            "label": {"lineName": f"Group_{i}"},
            "_raw_data": list(zip(x_pos, y, top, bot))
        })

    return all_lines_data

def generate_boxplot(ax, is_log_y, long_mode, h, dpi):
    num_boxes = random.randint(2, 6)
    
    data = []
    for _ in range(num_boxes):
        if is_log_y:
            sample = np.random.lognormal(mean=2.0, sigma=0.5, size=random.randint(20, 50))
        else:
            dice = random.random()
            if dice < 0.6:
                sample = np.random.normal(loc=50, scale=15, size=random.randint(20, 50))
            elif dice < 0.8:
                sample = np.random.uniform(10, 90, size=random.randint(20, 50))
            else:
                sample = np.random.gamma(2, 10, size=random.randint(20, 50)) + 10
        data.append(sample)

    stats = cbook.boxplot_stats(data)

    is_grayscale = random.random() < 0.20
    colors = ["black", "gray"] if is_grayscale else ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    boxprops = dict(linewidth=1.5)
    whiskerprops = dict(linewidth=1.5, linestyle='-')
    capprops = dict(linewidth=1.5)
    medianprops = dict(linewidth=2.0, color='red' if not is_grayscale else 'black')
    
    patch_artist = random.random() > 0.4
    
    ax.bxp(stats, patch_artist=patch_artist,
           boxprops=boxprops, whiskerprops=whiskerprops,
           capprops=capprops, medianprops=medianprops,
           showfliers=True)

    _raw_data = []
    for i, s in enumerate(stats):
        x_pos = i + 1
        median = s['med']
        whis_hi = s['whishi']
        whis_lo = s['whislo']
        
        top_err = whis_hi - median
        bot_err = median - whis_lo
        
        _raw_data.append((x_pos, median, top_err, bot_err))

    return [{
        "label": {"lineName": "Boxplot_Series"},
        "_raw_data": _raw_data
    }]

def calculate_labels(fig, ax, all_lines_data, h_inches, is_log_y):
    height_px = fig.canvas.get_width_height()[1]
    
    for line_data in all_lines_data:
        raw_points = line_data.pop("_raw_data")
        json_points = []
        
        for rx, ry, rt, rb in raw_points:
            disp = ax.transData.transform((rx, ry))
            px_y, px_x = float(height_px - disp[1]), float(disp[0])

            if is_log_y:
                disp_top = ax.transData.transform((rx, ry + rt))
                safe_bot = max(ry - rb, ax.get_ylim()[0] * 1.001)
                disp_bot = ax.transData.transform((rx, safe_bot))
            else:
                disp_top = ax.transData.transform((rx, ry + rt))
                disp_bot = ax.transData.transform((rx, ry - rb))

            px_top_loc = float(height_px - disp_top[1])
            px_bot_loc = float(height_px - disp_bot[1])

            d_top = abs(px_y - px_top_loc) if rt > 0 else 0
            d_bot = abs(px_y - px_bot_loc) if rb > 0 else 0

            max_safe = height_px * 1.5
            d_top = min(d_top, max_safe)
            d_bot = min(d_bot, max_safe)

            if d_top < MIN_PIXEL_NONZERO: d_top = 0.0
            if d_bot < MIN_PIXEL_NONZERO: d_bot = 0.0

            max_allowed = height_px * GEN_MAX_LABEL_PX_FACTOR
            d_top = min(d_top, max_allowed)
            d_bot = min(d_bot, max_allowed)

            json_points.append({
                "x": px_x,
                "y": px_y,
                "topBarPixelDistance": d_top,
                "bottomBarPixelDistance": d_bot
            })
        
        line_data["points"] = json_points
    
    return all_lines_data