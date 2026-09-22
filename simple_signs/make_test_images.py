import os
import random

from shapes import DRAW_FN, IMG_SIZE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_signs")

r = random.Random(42)

for kind in ("forward", "left", "stop"):
    img = DRAW_FN[kind](r, IMG_SIZE * 8)
    img.save(os.path.join(OUT_DIR, f"{kind}.png"))
    print(f"wrote {kind}.png")
