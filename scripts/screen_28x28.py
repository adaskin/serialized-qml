"""Screen of the 28x28 recipe: 3 vs 5, coherent, L=2, seed 42, one run with the given encoding scale / angle init / lr / batch.
The cosine schedule is that of a 100-epoch run (the lr is almost constant during the first epochs); stop it by hand after the epochs
you want to see. The per-epoch lines are the training accuracy; in epoch 1 every batch is new, so it is also a held-out accuracy.
Usage: python scripts/screen_28x28.py <enc_scale> <init_scale> <lr> <batch> [train_pc]   (results of the screens: data/screens/28x28/)"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
import train

enc, init, lr, batch = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
train.ENC_SCALE[28] = enc
train.RESULTS = Path("data/screens/28x28") / f"enc{enc:g}_init{init:g}_lr{lr:g}_b{batch}"
train.CKPT = train.RESULTS / "ckpt"
train.train(train.Run("3v5", 28, 2, "coherent", 42, batch=batch, init_scale=init, lr=lr,
                      train_pc=int(sys.argv[5]) if len(sys.argv) > 5 else 1000, eval_every=1000))
