# Raw data

The experiments use the standard MNIST archive in NumPy format. The binary archive is not committed; download and verify it from the repository root:

```bash
python src/download_mnist.py
```

- Source: `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`
- Destination: `data/raw/mnist.npz`
- SHA-256: `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`

All experimental outputs, including epoch-wise metrics and aggregated data, are tracked under `data/results/`. The training code expects arrays named `x_train`, `y_train`, `x_test`, and `y_test`.
