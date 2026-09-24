"""Train the sign locator + classifier CNNs, export sign images/font, and self-check on 640x480 frames.

Run on the PC:  python train.py          (train both, check, save)
                python train.py check    (only re-run the check on the saved models)
Produces:       sign_locator.keras, sign_classifier.keras, font.npy, test_signs/*.png
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import detect
from detect import CLASSES, IMG_SIZE, LOC_H, LOC_W
from signs import SIGNS, font_atlas, make_crop, make_scene, sign_card

L = tf.keras.layers
LOCATOR_SAMPLES = 14000
BACKGROUND_SHARE = 0.25
CLASSIFIER_PER_CLASS = 3000


# ---------------------------------------------------------------- locator

def locator_data(n, seed):
    rng = np.random.default_rng(seed)
    X = np.empty((n, LOC_H, LOC_W, 3), np.uint8)
    Y = np.zeros((n, 5), np.float32)  # [present, cx, cy, w, h], box normalised to 0..1
    for i in range(n):
        name = "background" if rng.random() < BACKGROUND_SHARE else SIGNS[rng.integers(len(SIGNS))]
        X[i], box = make_scene(name, rng, LOC_W, LOC_H)
        if box:
            x1, y1, x2, y2 = box
            Y[i] = [1, (x1 + x2) / 2 / LOC_W, (y1 + y2) / 2 / LOC_H, (x2 - x1) / LOC_W, (y2 - y1) / LOC_H]
    return X, Y


def locator_loss(y, p):
    present = y[:, 0]
    bce = tf.keras.losses.binary_crossentropy(y[:, :1], p[:, :1])
    box = tf.reduce_sum(tf.abs(y[:, 1:] - p[:, 1:]), axis=-1) * present  # box only counts if a sign is there
    return bce + 5.0 * box


def conv_stack(x, filters):
    for f in filters:
        x = L.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = L.MaxPooling2D()(x)
    return x


def build_locator():
    inp = L.Input((LOC_H, LOC_W, 3))
    x = L.Rescaling(1.0 / 255)(inp)
    x = conv_stack(x, [32, 32, 64, 64, 128])
    x = L.Flatten()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(128, activation="relu")(x)
    model = tf.keras.Model(inp, L.Dense(5, activation="sigmoid")(x))
    model.compile(optimizer="adam", loss=locator_loss)
    return model


# ---------------------------------------------------------------- classifier

def classifier_data(per_class, seed):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(CLASSES)), per_class)
    X = np.stack([make_crop(CLASSES[k], rng, IMG_SIZE) for k in y])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_classifier():
    inp = L.Input((IMG_SIZE, IMG_SIZE, 3))
    x = L.Rescaling(1.0 / 255)(inp)
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = conv_stack(x, [32, 64, 96, 128])
    x = L.Flatten()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(128, activation="relu")(x)
    model = tf.keras.Model(inp, L.Dense(len(CLASSES), activation="softmax")(x))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(model, X, Y, val):
    model.fit(X, Y, validation_data=val, epochs=30, batch_size=64, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])


# ---------------------------------------------------------------- end-to-end check

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def check(locator, classifier, per_class=40, seed=123):
    """Full 640x480 frames: right sign AND box overlapping it (IoU > 0.5); nothing on empty frames."""
    rng = np.random.default_rng(seed)
    hits = {}
    for name in CLASSES:
        ok = 0
        for _ in range(per_class):
            frame, box = make_scene(name, rng, 640, 480)
            det = detect.find_sign(frame, locator, classifier)
            if name == "background":
                ok += det is None
            else:
                ok += det is not None and det[1] == name and iou(det[0], box) > 0.5
        hits[name] = ok / per_class
        print(f"  {name:22s} {hits[name] * 100:5.1f}%")
    return hits


def main():
    if sys.argv[1:] == ["check"]:
        check(*detect.load_models())
        return

    train_locator = sys.argv[1:] != ["classifier"]  # `train.py classifier` retrains only that CNN

    np.save(detect.FONT_PATH, font_atlas())
    os.makedirs(os.path.join(detect.HERE, "test_signs"), exist_ok=True)
    for i, name in enumerate(SIGNS, 1):
        sign_card(name).save(os.path.join(detect.HERE, "test_signs", f"{i}_{name}.png"))

    if train_locator:
        print("Locator: generating scenes...")
        X, Y = locator_data(LOCATOR_SAMPLES, seed=0)
        val = locator_data(1500, seed=1)
        locator = build_locator()
        fit(locator, X, Y, val)
        del X, Y
        locator.save(detect.LOCATOR_PATH)
        print(f"Saved {detect.LOCATOR_PATH}")
    else:
        locator = tf.keras.models.load_model(detect.LOCATOR_PATH, compile=False)

    print("Classifier: generating crops...")
    X, y = classifier_data(CLASSIFIER_PER_CLASS, seed=2)
    val = classifier_data(200, seed=3)
    classifier = build_classifier()
    fit(classifier, X, y, val)
    _, acc = classifier.evaluate(*val, verbose=0)
    print(f"Classifier held-out crop accuracy: {acc * 100:.1f}%")

    classifier.save(detect.CLASSIFIER_PATH)
    print(f"Saved {detect.CLASSIFIER_PATH}")

    print("Full-frame detection check (640x480):")
    hits = check(locator, classifier)

    # Self-check: fail loudly if the models don't actually work (saved anyway, for debugging).
    assert acc > 0.95, f"classifier accuracy too low ({acc * 100:.1f}%)"
    assert min(hits.values()) >= 0.90, f"detection check failed: {hits}"
    print("Self-check passed.")


if __name__ == "__main__":
    main()
