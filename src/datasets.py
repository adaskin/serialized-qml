"""Generic loader for the extra benchmark datasets (tabular and image), used by src/bench.py and src/bench_classical.py.

Every dataset becomes the same thing the MNIST pipeline produces: a split with train / val / test arrays, the features in [0, 1], and
(for the circuit) a serialisation into patches of 4 values, one value per system qubit.

  tabular   features keep their file order (the serialisation order is part of the model, see reports/bench.md). They are clipped to
            the 1st-99th percentile of the training split and min-max scaled to [0, 1] (statistics from the training split only),
            then padded with zeros to a multiple of 4: T = ceil(F / 4) patches.
  images    grayscale, area-resized to 14x14, 2x2 patches in serpentine order (src/data.py): T = 49 patches.

Split (depends only on the dataset and the seed):
  test   fixed for a dataset (independent of the seed): stratified random draw, min(20% of the data, TEST_CAP), or the official test split.
  val    drawn per seed from the rest: min(10%, VAL_CAP) (monitoring only; the last epoch is always reported).
  train  per seed from the rest, at most TRAIN_CAP (default 10000) examples.
Tabular classes keep their natural frequencies (balanced accuracy is reported as well as accuracy); the image sets are class
balanced as in the MNIST grid (1000 / class for 10 classes).

Raw files live in data/raw/tier/ (or $QSB_RAW); `python src/datasets.py --check` prints what is there. Sources: UCI (waveform id 107,
wall-following robot id 194, covertype id 31, letter recognition id 59; second batch: spambase 94, default of credit card clients 350,
statlog landsat 146, dry bean 602), Fashion-MNIST (tf-keras-datasets mirror, see download_data.py), BloodMNIST and PneumoniaMNIST 28x28
(MedMNIST v2, zenodo record 10519652), Kuzushiji-MNIST (codh.rois.org/kmnist). BATCHES lists the first (A1) and second (A2) batch.
"""
import functools
import gzip
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import download_data
from data import to_patches

RAW = Path(os.environ.get("QSB_RAW", "data/raw/tier"))
TEST_CAP, VAL_CAP, TRAIN_CAP = 10000, 2000, 10000


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str                  # tabular | image
    n_classes: int
    n_features: int            # tabular: number of features; image: side after resizing
    note: str = ""

    @property
    def steps(self):           # number of patches T
        return -(-self.n_features // 4) if self.kind == "tabular" else (self.n_features // 2) ** 2


SPECS = {s.name: s for s in (
    Spec("waveform", "tabular", 3, 21, "UCI waveform (21 wave attributes; the 19 noise attributes of version 2 are dropped)"),
    Spec("wallrobot", "tabular", 4, 24, "UCI wall-following robot navigation, 24 ultrasound sensors in circular order"),
    Spec("covertype", "tabular", 7, 54, "UCI covertype, 10 quantitative + 4 wilderness + 40 soil indicators"),
    Spec("letter", "tabular", 26, 16, "UCI letter recognition, 16 integer features"),
    Spec("fashion", "image", 10, 14, "Fashion-MNIST, 14x14"),
    Spec("bloodmnist", "image", 8, 14, "BloodMNIST, grayscale, 14x14"),
    # second batch of Tier A
    Spec("spambase", "tabular", 2, 57, "UCI spambase, 48 word + 6 char frequencies + 3 capital-run statistics"),
    Spec("creditdefault", "tabular", 2, 23, "UCI default of credit card clients (23 features, integer codes kept as numbers)"),
    Spec("landsat", "tabular", 6, 36, "UCI statlog landsat satellite, 3x3 neighbourhood x 4 spectral bands (train + test files joined)"),
    Spec("drybean", "tabular", 7, 16, "UCI dry bean, 16 shape features"),
    Spec("pneumonia", "image", 2, 14, "PneumoniaMNIST 28x28 -> 14x14, official splits"),
    Spec("kmnist", "image", 10, 14, "Kuzushiji-MNIST 28x28 -> 14x14"),
)}
# review controls on MNIST (not part of the benchmark's SPECS): exactly the 8x8 split of the MNIST grid (src/data.py make_split,
# 1000 / 200 images per class, full test set), so the controls compare with the grid's runs of the same seed
MNIST_SPECS = {s.name: s for s in (
    Spec("mnist8_3v5", "mnist", 2, 8, "MNIST 3 vs 5, 8x8, the split of the MNIST grid"),
    Spec("mnist8_10class", "mnist", 10, 8, "MNIST 10 classes, 8x8, the split of the MNIST grid"),
)}
ALL_SPECS = {**SPECS, **MNIST_SPECS}
BATCHES = {"A1": ("waveform", "wallrobot", "covertype", "letter", "fashion", "bloodmnist"),
           "A2": ("spambase", "creditdefault", "landsat", "drybean", "pneumonia", "kmnist")}


def _find(sub, *patterns):
    for p in patterns:
        hits = sorted((RAW / sub).rglob(p)) if (RAW / sub).exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no {patterns} under {RAW / sub}: download the raw files first (see src/datasets.py)")


def _text(path):
    if path.suffix == ".gz":
        return io.StringIO(gzip.open(path, "rt").read())
    return open(path)


# ----------------------------------------------------------------------------- raw loaders: (X float32 (N, F), y int64) or images
def _waveform():
    rows = np.loadtxt(_text(_find("waveform", "waveform*.data*")), delimiter=",", dtype=np.float32)
    return rows[:, :21], rows[:, -1].astype(np.int64)


def _wallrobot():
    x, names = [], []
    for line in _text(_find("wallrobot", "sensor_readings_24.data")):
        *v, label = line.strip().split(",")
        x.append([float(t) for t in v])
        names.append(label)
    labels = sorted(set(names))
    return np.array(x, np.float32), np.array([labels.index(n) for n in names], np.int64)


def _covertype():
    rows = np.loadtxt(_text(_find("covertype", "covtype.data*")), delimiter=",", dtype=np.float32)
    return rows[:, :54], rows[:, 54].astype(np.int64) - 1


def _letter():
    x, y = [], []
    for line in _text(_find("letter", "letter-recognition.data")):
        if line.strip():
            c, *v = line.strip().split(",")
            y.append(ord(c) - ord("A"))
            x.append([float(t) for t in v])
    return np.array(x, np.float32), np.array(y, np.int64)


def _images(x):
    """(N, H, W[, 3]) uint8 -> (N, 14, 14) float in [0, 1] (luminance, area resize)."""
    x = torch.from_numpy(x).float() / 255.0
    if x.ndim == 4:
        x = (x * torch.tensor([0.299, 0.587, 0.114])).sum(-1)
    return F.interpolate(x[:, None], size=(14, 14), mode="area")[:, 0].numpy()


def _fashion():
    d = np.load(download_data.ensure("fashion"))
    return (_images(d["x_train"]), d["y_train"].astype(np.int64)), None, (_images(d["x_test"]), d["y_test"].astype(np.int64))


def _bloodmnist():
    d = np.load(_find("bloodmnist", "bloodmnist.npz"))
    get = lambda k: (_images(d[f"{k}_images"]), d[f"{k}_labels"].reshape(-1).astype(np.int64))
    return get("train"), get("val"), get("test")


def _spambase():
    rows = np.loadtxt(_text(_find("spambase", "spambase.data")), delimiter=",", dtype=np.float32)
    return rows[:, :57], rows[:, 57].astype(np.int64)


def _creditdefault():
    """data/raw/tier/creditdefault/creditdefault.csv: 23 feature columns then the label (0/1), no header and no ID column
    (converted from UCI's xls once; the xls itself is not read by the code)."""
    rows = np.loadtxt(_find("creditdefault", "creditdefault.csv"), delimiter=",", dtype=np.float32)
    return rows[:, :23], rows[:, 23].astype(np.int64)


def _landsat():
    rows = np.concatenate([np.loadtxt(_find("landsat", f"sat.{k}"), dtype=np.float32) for k in ("trn", "tst")])
    labels = sorted(set(rows[:, 36].astype(int)))
    return rows[:, :36], np.array([labels.index(int(v)) for v in rows[:, 36]], np.int64)


def _drybean():
    """data/raw/tier/drybean/drybean.csv: 16 feature columns then the class name, no header (converted from UCI's xlsx once)."""
    x, names = [], []
    for line in open(_find("drybean", "drybean.csv")):
        if line.strip():
            *v, label = line.strip().split(",")
            x.append([float(t) for t in v])
            names.append(label)
    labels = sorted(set(names))
    return np.array(x, np.float32), np.array([labels.index(n) for n in names], np.int64)


def _pneumonia():
    d = np.load(_find("pneumonia", "pneumoniamnist.npz"))
    get = lambda k: (_images(d[f"{k}_images"]), d[f"{k}_labels"].reshape(-1).astype(np.int64))
    return get("train"), get("val"), get("test")


def _kmnist():
    """data/raw/tier/kmnist/kmnist.npz with x_train, y_train, x_test, y_test (assembled once from the four kmnist-*.npz files)."""
    d = np.load(_find("kmnist", "kmnist.npz"))
    return (_images(d["x_train"]), d["y_train"].astype(np.int64)), None, (_images(d["x_test"]), d["y_test"].astype(np.int64))


LOADERS = {"waveform": _waveform, "wallrobot": _wallrobot, "covertype": _covertype, "letter": _letter,
           "fashion": _fashion, "bloodmnist": _bloodmnist, "spambase": _spambase, "creditdefault": _creditdefault,
           "landsat": _landsat, "drybean": _drybean, "pneumonia": _pneumonia, "kmnist": _kmnist}


@functools.lru_cache(maxsize=4)
def load(name):
    """tabular: (x, y); image: ((x_train, y_train), (x_val, y_val) or None, (x_test, y_test)) with the official splits."""
    return LOADERS[name]()


# ----------------------------------------------------------------------------- the split
def _stratified(y, n, rng):
    """n indices of y drawn at random with the class frequencies of y (at least one per class where possible)."""
    take = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, round(n * len(idx) / len(y)))
        take.append(rng.permutation(idx)[:k])
    return rng.permutation(np.concatenate(take))[:n]


def _balanced(y, per_class, rng):
    return rng.permutation(np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class] for c in np.unique(y)]))


def split_arrays(name, seed, train_cap=TRAIN_CAP):
    """(x_train, y_train, x_val, y_val, x_test, y_test) as float arrays in the original feature layout, unscaled (tabular) or in [0, 1] (image)."""
    spec = SPECS[name]
    if spec.kind == "tabular":
        x, y = load(name)
        fixed = np.random.default_rng(1234)                              # the test set never depends on the seed
        te = _stratified(y, min(int(0.2 * len(y)), TEST_CAP), fixed)
        rest = np.setdiff1d(np.arange(len(y)), te)
        rng = np.random.default_rng(seed)
        rest = rest[rng.permutation(len(rest))]
        n_val = min(int(0.1 * len(rest)), VAL_CAP)
        va, tr = rest[:n_val], rest[n_val:][:train_cap]
        return x[tr], y[tr], x[va], y[va], x[te], y[te]
    (xt, yt), val, (xe, ye) = load(name)
    rng = np.random.default_rng(seed)
    k = spec.n_classes
    tr = _balanced(yt, train_cap // k, rng)                              # class balanced, like the MNIST grid
    if val is None:                                                      # Fashion-MNIST has no official validation split: 200 / class from train
        rest = np.setdiff1d(np.arange(len(yt)), tr)
        va_idx = rest[_balanced(yt[rest], 200, rng)]
        xv, yv = xt[va_idx], yt[va_idx]
    else:
        va_idx = _balanced(val[1], VAL_CAP // k, rng)
        xv, yv = val[0][va_idx], val[1][va_idx]
    return xt[tr], yt[tr], xv, yv, xe, ye


def _scale(xtr, *others):
    lo, hi = np.percentile(xtr, 1, axis=0), np.percentile(xtr, 99, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    f = lambda a: np.clip((a - lo) / span, 0, 1).astype(np.float32)
    return [f(xtr)] + [f(a) for a in others]


def _mnist_split(name, seed, patches):
    """The split of the MNIST grid (src/train.py Run with the same seed), so review runs compare with the grid's runs."""
    from types import SimpleNamespace
    import data
    task = name.split("_", 1)[1]
    cfg = SimpleNamespace(dataset="mnist", classes=data.TASKS[task], image_size=MNIST_SPECS[name].n_features, patch=(2, 2), train_pc=1000,
                          val_pc=200, test_pc=None, seed=seed)
    return data.make_split(cfg, patches=patches)


def make_split(name, seed, patches=True, train_cap=TRAIN_CAP, perm=0):
    """Torch tensors (x_train, y_train, x_val, y_val, x_test, y_test). patches=True: x is (B, T, 4) for the circuit; patches=False:
    tabular (B, F) or image (B, 14, 14), the same numbers without the padding.
    perm > 0 (tabular only, review control of the serialisation order): the feature columns are put in a random order drawn with
    numpy seed `perm` (the same order for every training seed), before the padding and the grouping into patches."""
    spec = ALL_SPECS[name]
    if spec.kind == "mnist":
        assert perm == 0
        return _mnist_split(name, seed, patches)
    xtr, ytr, xva, yva, xte, yte = split_arrays(name, seed, train_cap)
    if spec.kind == "tabular":
        xtr, xva, xte = _scale(xtr, xva, xte)
        if perm:
            order = np.random.default_rng(perm).permutation(spec.n_features)
            xtr, xva, xte = (a[:, order] for a in (xtr, xva, xte))
    else:
        assert perm == 0
    xs = [torch.from_numpy(np.ascontiguousarray(a)) for a in (xtr, xva, xte)]
    ys = [torch.from_numpy(a) for a in (ytr, yva, yte)]
    if patches:
        if spec.kind == "tabular":
            pad = 4 * spec.steps - spec.n_features
            xs = [F.pad(a, (0, pad)).reshape(len(a), spec.steps, 4) for a in xs]
        else:
            xs = [to_patches(a, (2, 2)) for a in xs]
    return xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]


if __name__ == "__main__":
    import sys
    for n, s in SPECS.items():
        try:
            a = split_arrays(n, 42)
            counts = np.bincount(a[1], minlength=s.n_classes)
            print(f"{n:11s} T={s.steps:2d} train {len(a[1])} val {len(a[3])} test {len(a[5])} classes {s.n_classes} "
                  f"majority(test) {np.bincount(a[5]).max() / len(a[5]):.3f} train class counts {counts.min()}-{counts.max()}")
        except (FileNotFoundError, OSError) as e:
            print(f"{n:11s} missing: {str(e)[:100]}")
