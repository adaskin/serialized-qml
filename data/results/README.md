# Result files

One JSON file per training run, written by the script named for each folder. Nothing here is edited by hand. The tables and figures of the paper are generated from these files by `python scripts/paper_figures.py`, and the reports in `reports/` by the report scripts.

Seeds are 42, 43 and 44 unless stated otherwise. Unless stated otherwise, every run trains for 100 epochs, and `final` holds the last epoch.

## Folders

| folder | written by | contents | files |
|---|---|---|---|
| `quantum/` | `src/train.py` | the MNIST grid of the serialized circuit: tasks 3v5, 4v9, 4class, 10class; image size 8 (`s8`, 16 patches) and 14 (`s14`, 49 patches); memory `coherent` / `measured`; L = 2 / 4 | 96 |
| `classical/` | `src/classical.py` | MNIST baselines on the same splits: `logistic`, `mlp-L2/L4` and `cnn-L2/L4` (sized to the circuit with L = 2 / 4); three learning rates each (`lr0.001/0.003/0.01`, chosen afterwards by validation accuracy); sizes 8, 14, 28 | 540 |
| `amplitude/` | `src/amplitude.py` | MNIST amplitude-encoding circuits without serialization: `amp4` / `amp7` (4 or 7 qubits), L = 2 / 4 / 8; sizes 8, 14, 28 | 216 |
| `diagnostics/` | `src/train.py --set diagnostics` | the stalled 10-class 8x8 measured L2 seed-44 cell: images and initial angles separated by `initseedN` | 3 |
| `multibasis/` | `src/train.py --set multibasis`, `src/classical.py --set multibasis` | MNIST 10-class 8x8: circuit with the three-basis readout (`_rozzzxxxyyy`: Z, ZZ, X, XX, Y, YY, 42 features) and the MLP / CNN re-sized to it (`-f42`) | 21 |
| `bench/quantum/` | `src/bench.py` | the circuit on the twelve further datasets; see the suffixes below | 90 |
| `bench/classical/` | `src/bench_classical.py` | `logistic`, `mlp`, `cnn` (image sets) and gradient boosting `hgb` on the twelve datasets; `mlp_mb` is the MLP sized to the three-basis circuit and `mlp_head` the MLP sized to the circuit with an MLP head (Letter) | 333 |
| `bench/amplitude/` | `src/bench_amplitude.py` | amplitude encoding on 7 qubits (`amp7`), L = 4 / 8, on the twelve datasets | 72 |
| `review/quantum/` | `src/bench.py --set review` | controls asked for by the (mock) review; see the suffixes below | 110 |
| `review/classical/` | `src/bench_struct.py` | classical models with the structure of the circuit: `seqrnn` (site-dependent recurrent network), `mps` (matrix-product-state classifier), `hmm` (8-state input-driven hidden Markov model), each with three learning rates; `mlp` (seed 42 only, tabular sets) is the MLP rerun with stored predictions; on the twelve datasets and MNIST 8x8 3 vs 5 and 10 classes | 402 |
| `review/grad_variance.json` | `src/grad_variance.py` | gradient variance over 200 random initializations of the untrained circuit, one row per (n_mem, T, init); see below | 1 |

## File names of the circuit runs

`<dataset>_<memory>_L<layers>_<head>_seed<s>[suffix].json`, for example `letter_coherent_L4_linear_seed42.json`. MNIST grid files are `mnist_<task>_s<size>_<memory>_L<layers>_seed<s>.json`.

| suffix | meaning |
|---|---|
| (none) | the main run |
| `_e10` | 10-epoch pilot (seed 42); not used in the paper |
| `_rozzzxxxyyy` | three-basis readout (42 features) |
| `L9` in place of `L4` | depth control for the three-basis readout on Letter |
| `_mlp_` in place of `_linear_` | MLP head (64 units) instead of the linear layer, Letter only |
| `_tied` | one block reused at every patch; with `L16` / `L24` (4 x patches layers) it is the depth-matched tied block |
| `_product` | every CRot replaced by a Rot on its target (product state, same parameter count) |
| `_frozen` | block angles uniform in [-pi, pi) and frozen, only the linear head trained |
| `_perm1`, `_perm2`, `_perm3` | tabular features in a random order (numpy seed 1, 2, 3), seed 42 |
| `_pred` | rerun of the seed-42 main run that stores the test predictions (for McNemar tests) |

## Fields

Every run file has:
- `run`: the full configuration.
- `name`
- `params`: the parameter count.
- `seconds`: wall time.
- `final`: the last epoch.
- `history`: one row per epoch with `train_loss` and `train_acc`. Every 10 epochs a row also has `val_loss`, `val_acc`, `test_loss` and `test_acc`, and on the twelve datasets also `val_bacc` and `test_bacc` (balanced accuracy).

Some folders add fields:
- `quantum/` and `diagnostics/` add `grad_norm` per epoch (whole-model gradient norm before clipping) and `patch_grad_norm` (the gradient norm of every block W_i on the first batch of epochs 1 and 100).
- `bench/` and `review/` add `n_train`, `n_val` and `n_test`.
- The classical files there add `target_params` (the circuit size they are matched to) and `majority_test`.
- `review/` files add `test_pred` and `test_y` (predicted and true test labels, in test-set order). `review/quantum/` also adds `trainable` (the number of trained parameters, smaller for the frozen control).

`review/grad_variance.json` is a list of rows. Each row has:
- `n_mem` and `n` (= 4 + n_mem qubits).
- `T` (patches).
- `init`: `identity` = N(0, 0.08^2) as in training, or `uniform` = angles in [-pi, pi).
- `inits` (draws) and `batch` (inputs).
- `var_<cost>_<where>` and `meanabs_<cost>_<where>`, the variance and mean absolute value of the gradient. Here `<cost>` is `local` (Z on s0) or `global` (Z on all qubits), and `<where>` is `first` or `last`: one angle of W_1 or of W_T.

## Not committed

Raw datasets (`data/raw/`) and training checkpoints (`data/ckpt/`) are not in the repository; `src/data.py` and `src/datasets.py` give the sources. `data/screens/` holds the preliminary 28x28 screens described in `reports/caution_28x28.md`.
