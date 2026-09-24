"""Learned traffic-sign detection, TensorFlow + numpy only (no OpenCV). Runs on robot and PC.

Stage 1, locator CNN:    whole frame -> [sign present?, box cx, cy, w, h]
Stage 2, classifier CNN: crop of that box from the full-res frame -> which of the 6 signs
"""
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
LOCATOR_PATH = os.path.join(HERE, "sign_locator.keras")
CLASSIFIER_PATH = os.path.join(HERE, "sign_classifier.keras")
FONT_PATH = os.path.join(HERE, "font.npy")

CLASSES = ["background", "go_slow", "speed_up", "pedestrian_crossing",
           "road_closed", "u_turn_ahead", "roundabout_ahead"]
LABELS = {
    "go_slow": "Go Slow",
    "speed_up": "Speed Up",
    "pedestrian_crossing": "Pedestrian Crossing",
    "road_closed": "Road Closed",
    "u_turn_ahead": "U-turn Ahead",
    "roundabout_ahead": "Roundabout Ahead",
}
COLORS = {  # RGB box colours
    "go_slow": (255, 200, 0),
    "speed_up": (0, 120, 255),
    "pedestrian_crossing": (200, 0, 255),
    "road_closed": (255, 0, 0),
    "u_turn_ahead": (0, 255, 0),
    "roundabout_ahead": (0, 255, 255),
}
LOC_W, LOC_H = 160, 120  # locator input size
IMG_SIZE = 64            # classifier input size
OBJ_THRESHOLD = 0.5      # calibration knob: locator "sign present" probability
CONFIDENCE = 0.8         # calibration knob: classifier probability needed to report a sign
CROP_PAD = 1.15


def load_models():
    return (tf.keras.models.load_model(LOCATOR_PATH, compile=False),
            tf.keras.models.load_model(CLASSIFIER_PATH, compile=False))


def find_sign(frame, locator, classifier):
    """frame: HxWx3 uint8 RGB. Returns ((x1, y1, x2, y2), class_name, confidence) or None."""
    h, w = frame.shape[:2]
    small = tf.image.resize(frame[None].astype(np.float32), (LOC_H, LOC_W), antialias=True)
    obj, cx, cy, bw, bh = np.asarray(locator(small, training=False))[0]
    if obj < OBJ_THRESHOLD:
        return None
    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
    half = max(bw, bh) * CROP_PAD / 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = tf.image.resize(frame[None, y1:y2, x1:x2].astype(np.float32), (IMG_SIZE, IMG_SIZE), antialias=True)
    p = np.asarray(classifier(crop, training=False))[0]
    k = int(p.argmax())
    if k == 0 or p[k] < CONFIDENCE:
        return None
    box = (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2))
    return box, CLASSES[k], float(p[k])


# ---------------------------------------------------------------- drawing (numpy only)

_glyphs = None


def _font():
    global _glyphs
    if _glyphs is None:
        _glyphs = np.load(FONT_PATH).astype(bool)  # (95, H, W) bitmaps made by train.py
    return _glyphs


def put_text(img, text, x, y, fg, bg=None):
    """Draw text with its top-left at (x, y), clipped to the image."""
    g = _font()
    mask = np.concatenate([g[ord(c) - 32] if 32 <= ord(c) < 127 else g[0] for c in text], axis=1)
    H, W = img.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    mask = mask[:H - y, :W - x]
    region = img[y:y + mask.shape[0], x:x + mask.shape[1]]
    if bg is not None:
        region[:] = bg
    region[mask] = fg


def draw_box(img, box, color, t=3):
    H, W = img.shape[:2]
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(W, box[2]), min(H, box[3])
    img[y1:y1 + t, x1:x2] = color
    img[max(y1, y2 - t):y2, x1:x2] = color
    img[y1:y2, x1:x1 + t] = color
    img[y1:y2, max(x1, x2 - t):x2] = color


def draw(img, detection, top_margin=0):
    """Box + " Name 97% " label for a find_sign() result, drawn in place on an RGB image."""
    if detection is None:
        return img
    box, name, conf = detection
    color = COLORS[name]
    draw_box(img, box, color)
    th = _font().shape[1]
    y = box[1] - th - 2 if box[1] - th - 2 >= top_margin else box[3] + 2
    put_text(img, f" {LABELS[name]} {conf * 100:.0f}% ", box[0], y, (0, 0, 0), color)
    return img
