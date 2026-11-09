# Caution: the serialized circuit did not train on 28x28 images with 2x2 patches

The serialized circuit was run on 8x8 and 14x14 images (16 and 49 patches). On 28x28 images the same recipe has 196 patches and did not give a usable classifier, so 28x28 is not part of the grid. This note records what was tried, so that it can be reported as a limitation for patch-serialized models with small patches.

## Most patches are empty

2x2 patches of MNIST digits, first 20000 training images (`python scripts/empty_patches.py`):

| image | patches | pixels exactly 0 | patches with all 4 pixels 0 | patch positions empty in every image |
|---|---|---|---|---|
| 8x8 | 16 | 62.1% | 41.2% | 0 of 16 |
| 14x14 | 49 | 74.4% | 62.3% | 0 of 49 |
| 28x28 | 196 | 80.8% | 74.4% | 12 of 196 |

At 28x28 only about a quarter of the patches carry any image content. An empty patch applies a constant rotation (the encoding of a 0 pixel) and its own trainable gates, and the coherent memory has to carry the information of the active patches through all of them.

## What was run (MNIST 3 vs 5, coherent memory, L = 2, seed 42, Adam lr 0.01)

All numbers are training accuracy averaged over the epoch (the batches of epoch 1 are new to the model, so epoch 1 is also a held-out accuracy); validation and test accuracy only where stated. Chance is 0.50. The logs are in `data/screens/28x28/`; `scripts/screen_28x28.py` reproduces a screen.

**1. The grid settings** (pixel scale 0.08, gate angles initialised with standard deviation 0.08, batch 256, 1000 images per class, 8 updates per epoch):

| epoch | 10 | 20 | 26 |
|---|---|---|---|
| train | 0.508 | 0.606 | 0.705 |
| validation | 0.542 | 0.540 | - |
| test | 0.515 | 0.554 | - |

The run was stopped at epoch 26 of 100. It starts to fit the training set but validation accuracy stays at 0.54.

**2. Pixel scale with the same small init** (init 0.08, batch 256, 40 updates in 5 epochs): pixel scale 0.16, 0.33, 0.66 and 1.0 all gave training accuracy 0.52-0.53 at epoch 5 and validation 0.46-0.50 (chance).

**3. Larger initial gate angles** (batch 32, 62 updates per epoch, 2000 training images):

| pixel scale | init | lr | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|---|---|
| 0.08 | 0.08 | 0.01 | 0.495 | 0.483 | | |
| 0.08 | 0.08 | 0.03 | 0.475 | 0.496 | | |
| 0.03 | 0.08 | 0.01 | 0.512 | 0.547 | 0.505 | 0.500 |
| 0.08 | 0.3 | 0.01 | 0.514 | 0.608 | 0.680 | 0.707 |
| 0.16 | 0.3 | 0.01 | 0.506 | 0.487 | | |
| 0.08 | 0.6 | 0.01 | 0.689 | 0.709 | | |

Screens with init 1.0 and with pixel scales 0.04 and 0.16 at init 0.6 were stopped before their first epoch ended, when the decision was taken not to run 28x28 for the serialized model.

## Reading

- With the settings that work at 8x8 and 14x14 (pixel scale about 16 / number of patches, small gate angles at the start), the 28x28 circuit stays at chance for the first 80 updates and afterwards only fits the training set.
- Changing the pixel scale with the small init did not help. A larger starting size of the gate angles did: with init 0.6 the first epoch already reaches 0.69. None of these screens was followed to convergence, so it is not known whether a tuned recipe would reach a competitive accuracy.
- That most patches are empty is a plausible cause (the information of the active patches has to survive about 145 patches that add only gates) but it was not isolated. A direct test would be to pad 8x8 digits with empty border to 14x14 or 28x28 and compare accuracies; this was not run.
- For the write-up: patch-serialized circuits with small patches on sparse images (mostly empty patches) may not train with the recipe of the smaller images, so 28x28 results are not reported. The classical and the amplitude-encoding baselines are still reported at 28x28.

## Update 2026-11-04: 28x28 cancelled for the serialized model

The 28x28 column was briefly reopened (one seed, L = 2, angle init 0.6, otherwise the grid recipe) so that the 8x8 to 14x14 to 28x28 trend could be read, and cancelled the same day because the simulation is too slow. The runs shared the machine with the grid (10 runs) and other jobs, one core each. Time of the first epoch (including the one-off setup) and training accuracy after it:

| run (28x28, L=2, seed 42) | epoch 1 | train acc | 100 epochs would take |
|---|---|---|---|
| 3 vs 5, coherent | 2835 s | 0.623 | about 78 h |
| 4 vs 9, coherent | 2777 s | 0.599 | about 77 h |
| 10 classes, coherent | 7287 s | 0.388 | about 200 h |
| 3 vs 5, measured | 3724 s | 0.495 | about 104 h |

The 4-class coherent run had not finished an epoch when the job was stopped; the other measured runs had not started. No run reached epoch 10, so there is no validation accuracy, and no 28x28 result is reported for the serialized model. For the write-up: the simulation cost of the serialized circuit grows with steps x patches x gates, so 28x28 images with 2x2 patches were out of reach of a state-vector simulation here (days per run), and the screens above indicate that the small-init recipe does not train on mostly empty patches.
