"""MNIST loading and the fixed train / val / test split shared by the quantum and the classical models."""
import functools

import numpy as np
import torch
import torch.nn.functional as F

import download_data

TASKS = {"3v5": (3, 5), "4v9": (4, 9), "4class": (0, 1, 2, 3), "10class": tuple(range(10))}   # digits; label = position in the tuple


@functools.lru_cache(maxsize=4)
def load(dataset, size):
    d = np.load(download_data.ensure(dataset))
    out = {}
    for k in ("x_train", "x_test"):
        x = torch.from_numpy(d[k]).float() / 255.0
        out[k] = F.interpolate(x[:, None], size=(size, size), mode="area")[:, 0] if size != x.shape[-1] else x
    out["y_train"], out["y_test"] = torch.from_numpy(d["y_train"]), torch.from_numpy(d["y_test"])
    return out


def to_patches(images, patch):
    """(B,H,W) -> (B, T, ph*pw): non-overlapping patches in serpentine order."""
    b, h, w = images.shape
    ph, pw = patch
    gh, gw = h // ph, w // pw
    p = images.reshape(b, gh, ph, gw, pw).permute(0, 1, 3, 2, 4).reshape(b, gh, gw, ph * pw)
    cols = torch.arange(gw)
    order = torch.stack([(torch.arange(gw - 1, -1, -1) if r % 2 else cols) + r * gw for r in range(gh)]).reshape(-1)
    return p.reshape(b, gh * gw, ph * pw)[:, order]


def make_split(cfg, patches=True):
    """(x_train, y_train, x_val, y_val, x_test, y_test) for any object with dataset, classes, image_size, patch, train_pc,
    val_pc, test_pc (None = every test image of the class) and seed. Train and val images are disjoint draws from the MNIST
    training set, the test images come from the MNIST test set. x is (B, T, pixels per patch), or (B, H, W) if patches=False.
    The split depends only on the seed and the classes, so quantum and classical models see the same images."""
    d = load(cfg.dataset, cfg.image_size)
    g = torch.Generator().manual_seed(cfg.seed)
    parts = {k: ([], []) for k in ("tr", "va", "te")}
    for label, c in enumerate(cfg.classes):
        idx = (d["y_train"] == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        te = (d["y_test"] == c).nonzero().squeeze(1)
        te = te[torch.randperm(len(te), generator=g)][: cfg.test_pc]
        for key, ids, src in (("tr", idx[: cfg.train_pc], "x_train"), ("va", idx[cfg.train_pc: cfg.train_pc + cfg.val_pc], "x_train"),
                              ("te", te, "x_test")):
            parts[key][0].append(d[src][ids])
            parts[key][1].append(torch.full((len(ids),), label))
    out = []
    for key in ("tr", "va", "te"):
        x, y = torch.cat(parts[key][0]), torch.cat(parts[key][1])
        p = torch.randperm(len(y), generator=g)
        out += [to_patches(x[p], cfg.patch) if patches else x[p], y[p]]
    return out
