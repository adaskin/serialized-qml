#!/usr/bin/env python3
"""Download MNIST and Fashion-MNIST as npz files with keys x_train, y_train, x_test, y_test."""

import gzip
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DEST = Path("data/raw")


def mnist():
    path = DEST / "mnist.npz"
    if not path.exists():
        DEST.mkdir(parents=True, exist_ok=True)
        urlretrieve(BASE + "mnist.npz", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MNIST_SHA256:
        raise RuntimeError(f"MNIST checksum mismatch: {digest}")
    return path


def fashion():
    path = DEST / "fashion_mnist.npz"
    if path.exists():
        return path
    DEST.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, prefix in (("train", "train"), ("test", "t10k")):
        for kind, key, offset in (("images", "x", 16), ("labels", "y", 8)):
            gz = DEST / f"{prefix}-{kind}-idx{3 if kind == 'images' else 1}-ubyte.gz"
            if not gz.exists():
                urlretrieve(BASE + gz.name, gz)
            data = np.frombuffer(gzip.open(gz).read(), np.uint8, offset=offset)
            out[f"{key}_{split}"] = data.reshape(-1, 28, 28) if kind == "images" else data
    assert out["x_train"].shape == (60000, 28, 28) and out["y_test"].shape == (10000,)
    np.savez_compressed(path, **out)
    return path


def ensure(name):
    return {"mnist": mnist, "fashion": fashion}[name]()


if __name__ == "__main__":
    for name in sys.argv[1:] or ("mnist", "fashion"):
        print("ready:", ensure(name))
