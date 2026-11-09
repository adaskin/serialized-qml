# Serialized QML

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24059444.svg)](https://doi.org/10.5281/zenodo.24059444) 

simulation code for the paper:   
DASKIN, Ammar (2026). Serialized quantum classifiers decouple the number of parameters from the number of qubits. Zenodo. https://doi.org/10.5281/zenodo.24059444




A patch-serialized quantum image classifier written in PennyLane: one circuit class (`src/qcircuit.py`), native parameterized gates only, 7 qubits whatever the image size. The image is fed 2x2 pixels at a time (one pixel per qubit); every patch has its own trainable block W_t, so the number of parameters grows with the number of patches while the qubit count and the depth per patch stay fixed. The aim is a clear, scalable circuit whose parameter count can match classical models, not a quantum advantage. Design, parameter scaling and checks: [reports/circuit.md](reports/circuit.md).

## Experiment (one fixed grid)

| fixed | |
|---|---|
| data | MNIST, 1000 training images per class, 200 validation images per class (monitoring only), the full MNIST test set of the chosen digits |
| training | 100 epochs for every run, Adam lr 0.01 with cosine decay to 5%, batch 256, gradient clip 5, **no early stopping and no model selection: the last epoch is reported** |
| circuit | 2x2 patches, 4 pixel qubits + 3 ancilla qubits, xyz encoding, Z + ZZ readout, linear output layer; encoding scale 1 / 0.33 for 8x8 / 14x14 (about 16 / number of patches) |

| varied | |
|---|---|
| image size | 8x8 (16 patches), 14x14 (49); 28x28 (196 patches) is not run for the serialized circuit, see below |
| task | 3 vs 5, 4 vs 9, 4 classes (digits 0-3), 10 classes |
| memory | `coherent` (all qubits keep their state across patches) or `measured` (ancilla measured after every patch, only 3 classical bits connect the patches; system reloaded from \|0>) |
| layers | L = 2 or 4 repetitions of W_t per patch |
| seed | 42, 43, 44 |

That is 96 runs (`python src/train.py --list`). The runs are queued seed by seed (all of seed 42 first, then 43, then 44), so whenever the job is stopped the finished seeds form complete grids.

**28x28 is a caution, not a result.** With 2x2 patches a 28x28 digit has 196 patches and 74% of them are completely empty (62% at 14x14, 41% at 8x8). With the settings of the grid (encoding scale 0.08, gate angles initialised at 0.08) the circuit stays at chance or only fits the training set, and a larger angle initialisation (0.6) starts to learn. A 28x28 run of the grid was nevertheless started on 2026-11-04 (seed 42, L = 2, init 0.6) and cancelled the same day, because the simulation is too slow: about 47 minutes per epoch for 3 vs 5 and about 2 hours for 10 classes with coherent memory, and about 1 hour for 3 vs 5 with measured memory, on a machine shared with other jobs (3 to 8 days per 100-epoch run). The serialized model is therefore not reported at 28x28; the screens and timings are in [reports/caution_28x28.md](reports/caution_28x28.md). Classical and amplitude-encoding baselines are still run at 28x28.

## Diagnostics (not part of the 96-run grid)

- **Stalled run** (`python src/train.py --set diagnostics`): 10-class 8x8 measured L2 with seed 44 stalled at 47% training accuracy (seeds 42 and 43: 70% and 72%; there is no early stopping in the code and the gradient norms were normal). A plain rerun with seed 44 would reproduce it, because the seed sets both the images and the initial angles. `Run.init_seed` sets the angles separately: the same images with two other initialisations, and the seed-44 angles on the images of seed 42. Result (finished 2026-11-05): the seed-44 images stall with both other initialisations (test 50.9% and 54.6%), while the seed-42 images with the seed-44 angles reach 71.4% (seeds 42 and 43 with their own angles: 71.6%, 72.7%). So the stall comes with the seed-44 data (images or batch order, both follow `seed`), not with the initial angles. Results go to `data/results/diagnostics`; the report keeps the grid cell as it is (all three seeds) with a footnote.

Classical baselines on the same splits, epochs and optimizer (`src/classical.py`): logistic regression, an MLP and a small CNN, the last two sized to the parameter count of the circuit with L = 2 and with L = 4; seeds 42-44; the learning rate (0.001, 0.003, 0.01) is chosen by validation accuracy.

Amplitude-encoding baselines (`src/amplitude.py`): the whole image becomes the amplitude vector of 4 or 7 qubits (no serialization), followed by 2, 4 or 8 layers of the same Rot + CRot-chain block and the same readout; images larger than 2^n values are first shrunk (4 qubits: 4x4; 7 qubits: 8x8 as it is, 11x11 for 14x14 and 28x28). Same splits and training, seeds 42-44, 216 runs. Their parameter count does not grow with the image.

## Extra datasets (12 sets, not part of the grid)

The frozen circuit of the grid (coherent memory, L = 4, Z + ZZ readout, linear head, same training recipe, seeds 42-44, last epoch) on 12 more datasets: 8 tabular (Waveform, Wall-Following Robot, Covertype, Letter, Spambase, Credit Default, Landsat, Dry Bean; features scaled to [0, 1] and fed 4 at a time) and 4 image sets at 14x14 (Fashion-MNIST, BloodMNIST in grayscale, PneumoniaMNIST, Kuzushiji-MNIST; 49 patches). Training sets are capped at 10000 examples. Baselines on the same splits: logistic regression, MLP and CNN sized to the circuit, gradient boosting, amplitude encoding on 7 qubits (L = 4 and 8), and measured memory on the tabular sets. Setup, caveats and findings: [reports/bench.md](reports/bench.md); all tables: [reports/bench_results.md](reports/bench_results.md).

In short: the circuit is within about 2.5 points of a classical MLP of the same size on 8 of 12 sets, ahead on BloodMNIST and PneumoniaMNIST, and behind on Letter (26 classes) and Kuzushiji-MNIST; it beats amplitude encoding on the same 7 qubits on 11 of 12; coherent memory beats measured memory on 5 of 8 tabular sets (level on Dry Bean); gradient boosting is the strongest model on 8 of 12. A single head experiment (Letter, MLP head instead of the linear layer) is reported separately; every other run keeps the linear head.

**Multi-basis readout** (separate experiment): the same circuit read out in Z, ZZ, X, XX, Y, YY (42 features instead of 14; `readout="z+zz+x+xx+y+yy"`), on MNIST 10 classes 8x8, Letter, Covertype, Landsat and Dry Bean (Wall-Following and Waveform as small-class checks), against an MLP / CNN re-sized to the 42-feature circuit. It adds 8.9 points on Letter (a deeper Z + ZZ circuit with the same parameter count adds 0.9) and between -0.6 and +1.1 elsewhere; the re-sized classical models stay level or ahead. Table and figure: [reports/multibasis.md](reports/multibasis.md), `reports/figures/fig_multibasis.png`.

The raw files of the extra datasets are not committed (`data/raw/tier/`, sources in `src/datasets.py`); the result JSONs are.

## Review controls (not part of the grid)

Controls added after a (mock) journal review, on the same splits and recipe (results in `data/results/review/`):

- `python src/bench.py --set review` runs 110 circuit controls on Waveform, Wall-Following, Letter, Dry Bean, Landsat, Spambase and MNIST 8x8 (3 vs 5, 10 classes):
  - **tied blocks**: one W reused at every patch, plus a depth-matched tied block on the 4- to 6-patch sets
  - **product gates**: every CRot replaced by a Rot, so the state stays a product state with the same parameter count
  - **frozen random circuit**: only the linear head is trained
  - **three random feature orders**
  - **seed-42 reruns** that store test predictions for McNemar tests
- `python src/bench_struct.py` runs 402 classical models with the structure of the circuit: a site-dependent RNN, an MPS classifier and an 8-state input-driven HMM, plus the MLP rerun with predictions.
- `python src/grad_variance.py` computes the gradient variance over 200 random initializations for 5 to 11 qubits (1 to 7 memory qubits), 4 and 16 patches, local and global cost, near-identity and uniform initialization.

Report with all tables and the gradient-variance figure: [reports/review.md](reports/review.md) (`python src/review_report.py`).

In short:
- Every control lowers the accuracy, and the frozen circuit most of all.
- From uniform random angles the gradient variance halves with every qubit.
- From the near-identity start used in training, with a local cost, the variance stays flat from 5 to 11 qubits.

## Run

```bash
python -m pip install -r requirements.txt
python src/train.py --workers 10          # the grid; finished runs are skipped, interrupted runs resume from data/ckpt
python src/classical.py --workers 4       # classical baselines
python src/amplitude.py --workers 4       # amplitude-encoding baselines
python src/report.py                      # reports/results.md and reports/figures/fig_*.png
python src/train.py --set multibasis      # MNIST 10-class 8x8 L4 with the multi-basis readout (classical.py --set multibasis: re-sized baselines)

# extra datasets (raw files in data/raw/tier/)
python src/bench.py --set main            # circuits; also --set measured | head | readout | control | multibasis, --batch A1 | A2
python src/bench_classical.py --workers 4 # logistic, MLP, CNN, gradient boosting
python src/bench_amplitude.py --workers 4 # amplitude encoding, 7 qubits, L = 4 and 8
python src/bench_report.py                # reports/bench_results.md
python src/multibasis_report.py           # reports/multibasis.md and reports/figures/fig_multibasis.png
python src/bench.py --set review         # review controls (tied, product, frozen, feature order, prediction reruns)
python src/bench_struct.py --workers 8   # classical models with the structure of the circuit
python src/grad_variance.py --inits 200  # gradient variance vs number of qubits
python src/review_report.py              # reports/review.md and reports/figures/fig_gradvar.png
python scripts/paper_figures.py          # figures and tables of the paper (paper/figures, paper/tables)
python -m pytest tests
```

Cost: one core per run; measured-memory runs are about 4-5 times as expensive as coherent ones and make up three quarters of the grid (the 14x14 10-class measured runs with 4 layers take several days each on one core). With 10 workers on 8 cores expect about 7-9 days for all three seeds. A run needs 1-3 GB of RAM with a CPU build of torch; `train.py` only starts runs while their estimated RAM fits in `--ram-gb` and the machine has `--min-free-gb` available.

## Layout

```text
src/qcircuit.py     SerializedQC: the circuit (coherent or measured memory; readout Z + ZZ or more Pauli bases)
src/data.py         MNIST loading, patches, the train / val / test split shared by all models
src/train.py        the grid of circuit runs, resumable training
src/classical.py    classical baselines
src/amplitude.py    amplitude-encoding quantum baselines
src/report.py       tables and figures
src/datasets.py     the 12 extra datasets: loading, split, scaling, patches
src/bench*.py       extra-dataset runs (bench.py circuits, bench_classical.py, bench_amplitude.py) and their report
src/multibasis_report.py   multi-basis readout table and figure
src/bench_struct.py structure-matched classical baselines (RNN, MPS, HMM)
src/grad_variance.py  gradient variance over random initializations
scripts/            draw_circuit.py (figures of the circuit), expressivity.py (Jacobian rank), paper_figures.py (figures and tables of the paper)
tests/              circuit checks (gradients, density-matrix equivalence of the measured memory), pipeline checks
data/results/       one JSON per run (quantum/, classical/, amplitude/, bench/, multibasis/, review/); what each folder holds: data/results/README.md; data/raw has the downloads (not committed)
paper/              the manuscript (main.tex with supplementary.tex)
```

## Citation
If you use this code or the result files, please cite  the paper:  
DASKIN, A. (2026). Serialized quantum classifiers decouple the number of parameters from the number of qubits. Zenodo. https://doi.org/10.5281/zenodo.24059444


