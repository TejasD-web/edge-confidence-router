"""Draws docs/confidence_lo_002.png: the lo_002 photo with the two numbers marked. Run from the repo root."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/numbers/images/lo_002.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/confidence_lo_002.png"
SURF, INK, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#2a78d6", "#eb6834"
img = Image.open(SRC).convert("RGB")
fig = plt.figure(figsize=(9.6, 7.2), dpi=100, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
ax.add_patch(Rectangle((-1, 207), 47, 24, fill=False, ec=ORANGE, lw=3))
ax.add_patch(Rectangle((416, 209), 30, 22, fill=False, ec=BLUE, lw=3))
kw = dict(fontsize=15, fontweight="bold", color=INK)
ax.annotate("neighbouring locomotive: 37420", xy=(22, 207), xytext=(8, 150), **kw,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=2.5),
            bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=ORANGE, lw=2))
ax.annotate("the target: 47617", xy=(431, 231), xytext=(380, 300), **kw,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=2.5),
            bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=BLUE, lw=2))
fig.savefig(OUT, facecolor=SURF)
