import uuid
import random
from matplotlib import pyplot as plt
from config import PLOT_TYPE_PROBS

def generate_image(output_dir: str):
  """Generates one image"""

  id = str(uuid.uuid4())

  w = random.randint(8, 15)
  h = random.randint(6, 12)
  dpi = random.choice([72, 100, 150, 200])
  fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

  dice = random.random()
  if dice < PLOT_TYPE_PROBS["linegraph"]:
    data = generate_linegraph(ax)
  elif dice < PLOT_TYPE_PROBS["linegraph"] + PLOT_TYPE_PROBS["barchart"]:
    data = generate_barchart(ax)
  else:
    data = generate_boxplot(ax)

  fig.savefig(f"{output_dir}/images/{id}.png")
  plt.close(fig)
  # TODO: Save data to file

def generate_linegraph(ax):
  pass

def generate_barchart(ax):
  pass

def generate_boxplot(ax):
  pass