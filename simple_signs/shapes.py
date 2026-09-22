import math

import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 64
CLASSES = ["background", "forward", "left", "stop"]


def _rand_bg_color(r):
    return tuple(r.randint(180, 255) for _ in range(3))


def draw_forward(r, size):
    img = Image.new("RGB", (size, size), _rand_bg_color(r))
    d = ImageDraw.Draw(img)
    c = size // 2
    rad = int(size * r.uniform(0.30, 0.42))
    color = (r.randint(0, 60), r.randint(40, 100), r.randint(150, 255))
    d.ellipse([c - rad, c - rad, c + rad, c + rad], fill=color)

    a = rad * 0.55
    d.polygon([
        (c, c - a), (c - a * 0.6, c - a * 0.1), (c - a * 0.25, c - a * 0.1),
        (c - a * 0.25, c + a * 0.6), (c + a * 0.25, c + a * 0.6),
        (c + a * 0.25, c - a * 0.1), (c + a * 0.6, c - a * 0.1),
    ], fill="white")
    return img


def draw_left(r, size):
    img = Image.new("RGB", (size, size), _rand_bg_color(r))
    d = ImageDraw.Draw(img)
    c = size // 2
    rad = int(size * r.uniform(0.30, 0.42))
    color = (r.randint(0, 60), r.randint(40, 100), r.randint(150, 255))
    d.ellipse([c - rad, c - rad, c + rad, c + rad], fill=color)

    a = rad * 0.55
    d.polygon([
        (c - a, c), (c - a * 0.1, c - a * 0.6), (c - a * 0.1, c - a * 0.25),
        (c + a * 0.6, c - a * 0.25), (c + a * 0.6, c + a * 0.25),
        (c - a * 0.1, c + a * 0.25), (c - a * 0.1, c + a * 0.6),
    ], fill="white")
    return img


def draw_stop(r, size):
    img = Image.new("RGB", (size, size), _rand_bg_color(r))
    d = ImageDraw.Draw(img)
    c = size // 2
    rad = size * r.uniform(0.30, 0.42)
    color = (r.randint(150, 255), r.randint(0, 50), r.randint(0, 50))
    pts = [
        (c + rad * math.cos(math.pi / 8 + i * math.pi / 4),
         c + rad * math.sin(math.pi / 8 + i * math.pi / 4))
        for i in range(8)
    ]
    d.polygon(pts, fill=color)
    return img


def draw_background(r, size):
    if r.random() < 0.5:
        arr = np.random.RandomState(r.randint(0, 1_000_000)).randint(
            0, 256, (size, size, 3), dtype=np.uint8
        )
        return Image.fromarray(arr, "RGB")
    return Image.new("RGB", (size, size), _rand_bg_color(r))


DRAW_FN = {
    "background": draw_background,
    "forward": draw_forward,
    "left": draw_left,
    "stop": draw_stop,
}


def make_sample(kind, r):
    size = IMG_SIZE * 2
    img = DRAW_FN[kind](r, size)
    img = img.rotate(r.uniform(-20, 20), fillcolor=_rand_bg_color(r))
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img).astype(np.float32)
    if r.random() < 0.7:
        arr = arr * r.uniform(0.7, 1.3)
    arr += np.random.RandomState(r.randint(0, 1_000_000)).normal(0, 8, arr.shape)
    return np.clip(arr, 0, 255)
