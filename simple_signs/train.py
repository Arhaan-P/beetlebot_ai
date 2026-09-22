import random

import numpy as np
import tensorflow as tf

from shapes import CLASSES, IMG_SIZE, make_sample

SAMPLES_PER_CLASS = 250
MODEL_PATH = "simple_sign_model.h5"


def build_dataset(samples_per_class, seed):
    r = random.Random(seed)
    X, y = [], []
    for label, kind in enumerate(CLASSES):
        for _ in range(samples_per_class):
            X.append(make_sample(kind, r))
            y.append(label)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    idx = np.random.RandomState(seed).permutation(len(X))
    return X[idx], y[idx]


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES)),
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    X_train, y_train = build_dataset(SAMPLES_PER_CLASS, seed=0)
    X_val, y_val = build_dataset(40, seed=1)

    model = build_model()
    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=2)

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Held-out accuracy: {acc * 100:.1f}%")

    # Self-check: refuse to ship a model that didn't actually learn the shapes.
    assert acc > 0.90, f"Model accuracy too low ({acc * 100:.1f}%), not saving"

    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
