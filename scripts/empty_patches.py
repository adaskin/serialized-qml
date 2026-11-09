"""Share of empty 2x2 patches of MNIST at the three image sizes (first 20000 training images). Usage: python scripts/empty_patches.py"""
import sys

sys.path.insert(0, "src")
from data import load, to_patches

print("| image | patches | pixels exactly 0 | patches with all 4 pixels 0 | patch positions empty in every image |\n|---|---|---|---|---|")
for size in (8, 14, 28):
    x = load("mnist", size)["x_train"][:20000]
    p = to_patches(x, (2, 2))
    empty = p.amax(-1) == 0
    print(f"| {size}x{size} | {p.shape[1]} | {(x == 0).float().mean():.1%} | {empty.float().mean():.1%} | {int(empty.all(0).sum())} of {p.shape[1]} |")
