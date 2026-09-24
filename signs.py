"""Draws the 6 traffic signs and synthesises training images (scenes + crops). PC-side only, uses Pillow."""
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from detect import CLASSES

S = 256  # master sign resolution
RED = (200, 25, 35, 255)
BLUE = (15, 80, 180, 255)
WHITE = (255, 255, 255, 255)
BLACK = (15, 15, 15, 255)


def _font(size, names=("arialbd.ttf", "DejaVuSans-Bold.ttf")):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _scale(pts, c, k):
    return [(c[0] + (x - c[0]) * k, c[1] + (y - c[1]) * k) for x, y in pts]


def _arrowhead(d, p, t, length, half_w):
    n = (-t[1], t[0])
    d.polygon([
        (p[0] + t[0] * length, p[1] + t[1] * length),
        (p[0] + n[0] * half_w, p[1] + n[1] * half_w),
        (p[0] - n[0] * half_w, p[1] - n[1] * half_w),
    ], fill=BLACK)


def _warning_triangle(d):
    """Red-bordered white triangle. Returns (cx, cy, u): content centre and size unit."""
    side = 0.96 * S
    h = side * math.sqrt(3) / 2
    top = (S - h) / 2
    pts = [(S / 2, top), (S / 2 - side / 2, top + h), (S / 2 + side / 2, top + h)]
    c = (S / 2, top + 2 * h / 3)
    r = h / 3  # inradius
    border = 0.085 * S
    d.polygon(_scale(pts, c, (r + 0.02 * S) / r), fill=WHITE)
    d.polygon(pts, fill=RED)
    d.polygon(_scale(pts, c, (r - border) / r), fill=WHITE)
    return c[0], c[1], r - border


def _slow(d, cx, cy, u):
    d.text((cx, cy + 0.12 * u), "SLOW", font=_font(int(0.62 * u)), fill=BLACK, anchor="mm")


def _u_turn(d, cx, cy, u):
    r, lw, top = 0.42 * u, int(0.22 * u), cy - 0.1 * u
    d.line([(cx + r, cy + 0.75 * u), (cx + r, top)], fill=BLACK, width=lw)
    o = r + lw / 2
    d.arc([cx - o, top - o, cx + o, top + o], 180, 360, fill=BLACK, width=lw)
    d.line([(cx - r, top), (cx - r, cy + 0.3 * u)], fill=BLACK, width=lw)
    _arrowhead(d, (cx - r, cy + 0.3 * u), (0, 1), 0.45 * u, 0.34 * u)


def _roundabout(d, cx, cy, u):
    cy += 0.08 * u
    R, lw = 0.56 * u, int(0.2 * u)
    o = R + lw / 2
    for k in range(3):
        start = -90 + 120 * k + 15
        end = start + 75
        d.arc([cx - o, cy - o, cx + o, cy + o], start, end, fill=BLACK, width=lw)
        e = math.radians(end)
        # PIL angles run clockwise on screen, so the tangent at e is (-sin, cos).
        _arrowhead(d, (cx + R * math.cos(e), cy + R * math.sin(e)),
                   (-math.sin(e), math.cos(e)), 0.36 * u, 0.3 * u)


def _pedestrian_board(d, blank):
    d.rounded_rectangle([0.04 * S, 0.04 * S, 0.96 * S, 0.96 * S], 0.06 * S, fill=WHITE)
    d.rounded_rectangle([0.07 * S, 0.07 * S, 0.93 * S, 0.93 * S], 0.05 * S, fill=BLUE)
    side = 0.74 * S
    h = side * math.sqrt(3) / 2
    top = (S - h) / 2
    d.polygon([(S / 2, top), (S / 2 - side / 2, top + h), (S / 2 + side / 2, top + h)], fill=WHITE)
    if blank:
        return
    cx, cy, u = S / 2, top + 2 * h / 3, h / 3
    lw = int(0.16 * u)
    d.ellipse([cx - 0.15 * u, cy - 0.72 * u, cx + 0.15 * u, cy - 0.42 * u], fill=BLACK)
    hip = (cx - 0.05 * u, cy + 0.25 * u)
    d.line([(cx + 0.02 * u, cy - 0.38 * u), hip], fill=BLACK, width=lw)
    d.line([hip, (cx - 0.42 * u, cy + 0.82 * u)], fill=BLACK, width=lw)
    d.line([hip, (cx + 0.32 * u, cy + 0.82 * u)], fill=BLACK, width=lw)
    d.line([(cx - 0.38 * u, cy + 0.05 * u), (cx, cy - 0.25 * u), (cx + 0.36 * u, cy + 0.02 * u)],
           fill=BLACK, width=lw, joint="curve")


def _disc(d, color, inner=None):
    d.ellipse([0.02 * S, 0.02 * S, 0.98 * S, 0.98 * S], fill=WHITE)
    d.ellipse([0.04 * S, 0.04 * S, 0.96 * S, 0.96 * S], fill=color)
    if inner:
        d.ellipse([0.15 * S, 0.15 * S, 0.85 * S, 0.85 * S], fill=inner)


def draw_sign(name, blank=False):
    """RGBA master image of a sign; blank=True leaves out the symbol (used as a negative example)."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if name in ("go_slow", "u_turn_ahead", "roundabout_ahead"):
        cx, cy, u = _warning_triangle(d)
        if not blank:
            {"go_slow": _slow, "u_turn_ahead": _u_turn, "roundabout_ahead": _roundabout}[name](d, cx, cy, u)
    elif name == "speed_up":
        _disc(d, BLUE)
        if not blank:
            for y in (0.40 * S, 0.62 * S):
                d.line([(0.28 * S, y + 0.1 * S), (0.5 * S, y - 0.1 * S), (0.72 * S, y + 0.1 * S)],
                       fill=WHITE, width=int(0.08 * S), joint="curve")
    elif name == "road_closed":
        _disc(d, RED, inner=WHITE)
        if not blank:
            f = _font(int(0.13 * S))
            d.text((S / 2, 0.42 * S), "ROAD", font=f, fill=BLACK, anchor="mm")
            d.text((S / 2, 0.58 * S), "CLOSED", font=f, fill=BLACK, anchor="mm")
    elif name == "pedestrian_crossing":
        _pedestrian_board(d, blank)
    else:
        raise ValueError(name)
    return img


SIGNS = [n for n in CLASSES if n != "background"]
_MASTER = {n: draw_sign(n) for n in SIGNS}
_BLANK = {n: draw_sign(n, blank=True) for n in SIGNS}


def sign_card(name, size=600):
    """Sign on a white card, as it should be shown on a phone / printed (RGB)."""
    card = Image.new("RGB", (size, size), "white")
    m = int(size * 0.08)
    sign = _MASTER[name].resize((size - 2 * m,) * 2, Image.LANCZOS)
    card.paste(sign, (m, m), sign)
    return card


def font_atlas(height=20):
    """Bitmap glyphs for ASCII 32..126, shape (95, H, W) uint8 0/1, so the robot can draw text with numpy."""
    font = _font(height, ("consolab.ttf", "DejaVuSansMono-Bold.ttf"))
    asc, desc = font.getmetrics()
    w = int(font.getlength("M")) + 1
    glyphs = []
    for c in range(32, 127):
        img = Image.new("L", (w, asc + desc), 0)
        ImageDraw.Draw(img).text((0, 0), chr(c), font=font, fill=255)
        glyphs.append(np.asarray(img) > 100)
    return np.stack(glyphs).astype(np.uint8)


# ---------------------------------------------------------------- synthesis

def _rand_color(rng):
    return tuple(int(c) for c in rng.integers(0, 256, 3))


def _background(rng, w, h):
    kind = rng.integers(4)
    if kind == 0:
        img = Image.new("RGB", (w, h), _rand_color(rng))
    elif kind == 1:
        a, b = np.array(_rand_color(rng)), np.array(_rand_color(rng))
        t = np.linspace(0, 1, w)[None, :, None] if rng.random() < 0.5 else np.linspace(0, 1, h)[:, None, None]
        img = Image.fromarray(np.broadcast_to(a + (b - a) * t, (h, w, 3)).astype(np.uint8))
    elif kind == 2:
        small = rng.integers(0, 256, (rng.integers(2, 12), rng.integers(2, 12), 3)).astype(np.uint8)
        img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    else:
        img = Image.new("RGB", (w, h), (int(rng.integers(190, 256)),) * 3)  # wall / white paper
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(0, 7)):  # clutter, including sign-coloured blobs
        col = [(200, 30, 30), (20, 70, 190)][rng.integers(2)] if rng.random() < 0.3 else _rand_color(rng)
        x1, x2 = sorted(rng.integers(0, w, 2))
        y1, y2 = sorted(rng.integers(0, h, 2))
        shape = rng.integers(3)
        if shape == 0:
            d.rectangle([x1, y1, x2, y2], fill=col)
        elif shape == 1:
            d.ellipse([x1, y1, x2, y2], fill=col)
        else:
            d.line([x1, y1, x2, y2], fill=col, width=int(rng.integers(1, max(2, w // 60))))
    return img.convert("RGBA")


def _perspective_coeffs(dst, src):
    """Coefficients for Image.transform(PERSPECTIVE): maps output points dst -> input points src."""
    A, B = [], []
    for (x, y), (X, Y) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -X * x, -X * y], [0, 0, 0, x, y, 1, -Y * x, -Y * y]]
        B += [X, Y]
    return np.linalg.solve(np.array(A, float), np.array(B, float))


def _paste(bg, sign, rng, size, center, card):
    """Perspective-warp sign (optionally on a white card / phone) onto bg in place. Returns sign bbox."""
    w, h = bg.size
    ang = math.radians(rng.uniform(-12, 12))
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
    jitter = rng.uniform(-0.07, 0.07, (4, 2)) * size
    src = [(0, 0), (S, 0), (S, S), (0, S)]

    def warp(img, k):
        dst = (corners * k + jitter) @ rot.T + center
        return img.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(dst, src), Image.BILINEAR)

    if card:
        k = rng.uniform(1.05, 1.3)
        if rng.random() < 0.4:  # phone bezel around the card
            bg.alpha_composite(warp(Image.new("RGBA", (S, S), (20, 20, 20, 255)), k * 1.12))
        bg.alpha_composite(warp(Image.new("RGBA", (S, S), (int(rng.integers(200, 256)),) * 3 + (255,)), k))
    warped = warp(sign, 1.0)
    bg.alpha_composite(warped)
    return warped.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()


def _photometric(img, rng):
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5) * img.size[0] / 160))
    a = np.asarray(img, np.float32)
    a = a * rng.uniform(0.6, 1.3) + rng.uniform(-40, 40)
    a = a * rng.uniform(0.8, 1.2, 3)
    if rng.random() < 0.2:
        a = a + (a.mean(axis=2, keepdims=True) - a) * rng.uniform(0, 0.5)
    a = 255 * (np.clip(a, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    a = np.clip(a + rng.normal(0, rng.uniform(0, 10), a.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=int(rng.integers(30, 85)))
        a = np.asarray(Image.open(buf).convert("RGB"))
    return a


def make_scene(name, rng, w=160, h=120):
    """Camera-like frame. Returns (RGB uint8 array, sign box (x1, y1, x2, y2) in px, or None).

    Rendered at 2x then downscaled, like a real camera frame resized for the locator."""
    W, H = 2 * w, 2 * h
    bg = _background(rng, W, H)
    box = None
    if name == "background":
        if rng.random() < 0.3:  # sign shape without its symbol is not a sign
            size = rng.uniform(0.2, 0.7) * H
            c = rng.uniform([size / 2, size / 2], [W - size / 2, H - size / 2])
            _paste(bg, _BLANK[SIGNS[rng.integers(len(SIGNS))]], rng, size, c, rng.random() < 0.6)
    else:
        size = rng.uniform(0.19, 0.75) * H
        c = rng.uniform([size / 2, size / 2], [W - size / 2, H - size / 2])
        box = tuple(v / 2 for v in _paste(bg, _MASTER[name], rng, size, c, rng.random() < 0.6))
    img = bg.convert("RGB").resize((w, h), Image.BOX)
    return _photometric(img, rng), box


def make_crop(name, rng, size=64):
    """Classifier input: a crop of a full make_scene() frame around the sign, resized to `size`.

    Built from a full-resolution scene (not a small canvas of its own) so the same blur/noise/JPEG
    photometric pipeline that degrades a real camera crop also degrades this training crop -- a
    small canvas blurred directly at crop scale looks much cleaner than a small sign in a big blurry
    frame, which used to make the classifier fail on real (locator-produced) crops it had never
    effectively seen during training."""
    W, H = 320, 240  # a stand-in "real" frame size (all relative; only the crop around the sign is kept)
    if name == "background":
        frame, box = make_scene("background", rng, W, H)
        if box is None and rng.random() < 0.5:
            # no sign in this frame either; take a crop of plain background/clutter
            side = rng.uniform(0.15, 0.5) * min(W, H)
            cx, cy = rng.uniform([side, side], [W - side, H - side])
            box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        elif box is None:
            box = (W / 2 - size / 2, H / 2 - size / 2, W / 2 + size / 2, H / 2 + size / 2)
    else:
        frame, box = make_scene(name, rng, W, H)
        while box is None:  # make_scene always places a sign for a real class; guards a future change
            frame, box = make_scene(name, rng, W, H)

    # detect.py pads the locator's box by CROP_PAD (~1.15x) before classifying; mimic that padding
    # and the locator's imperfect localisation with extra jitter/scale around the ground-truth box.
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(bw, bh) * rng.uniform(0.95, 1.35) / 2
    cx += rng.uniform(-0.1, 0.1) * half
    cy += rng.uniform(-0.1, 0.1) * half
    cx1, cy1 = int(max(0, cx - half)), int(max(0, cy - half))
    cx2, cy2 = int(min(W, cx + half)), int(min(H, cy + half))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        cx1, cy1, cx2, cy2 = 0, 0, W, H
    crop = Image.fromarray(frame[cy1:cy2, cx1:cx2])
    return np.asarray(crop.resize((size, size), Image.BOX))
