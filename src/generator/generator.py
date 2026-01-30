import uuid
import random
from matplotlib import pyplot as plt
from config import DPI_OPTIONS, PLOT_TYPE_PROBS

def generate_image():
  """Generates one image"""
  id = str(uuid.uuid4())
  w = random.randint(8, 15)
  h = random.randint(6, 12)
  dpi = random.choice(DPI_OPTIONS)
  fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

  set_background(ax)
  set_spines(ax)

  dice = random.random()
  if dice < PLOT_TYPE_PROBS["linegraph"]:
    data = generate_linegraph(ax)
  elif dice < PLOT_TYPE_PROBS["linegraph"] + PLOT_TYPE_PROBS["barchart"]:
    data = generate_barchart(ax)
  else:
    data = generate_boxplot(ax)

  return fig, data

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

def generate_linegraph(ax):
  pass

def generate_barchart(ax):
  pass

def generate_boxplot(ax):
  pass