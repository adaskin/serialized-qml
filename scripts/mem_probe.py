"""Peak RAM and time of one forward + backward step of SerializedQC (one core).
Usage: python scripts/mem_probe.py <image_size> <layers> <batch> <coherent|measured> [segment]
Also prints the library versions, because RAM per run differed a lot between machines."""
import os
import resource
import sys
import time

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
sys.path.insert(0, "src")

import numpy
import pennylane
import torch

from qcircuit import SerializedQC

size, layers, batch, memory = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
segment = int(sys.argv[5]) if len(sys.argv) > 5 else 8
torch.set_num_threads(1)
steps = (size // 2) ** 2
model = SerializedQC(n_steps=steps, layers=layers, n_out=10, memory=memory, enc_scale=1.0, segment=segment)
x = torch.rand(batch, steps, 4)
for i in range(2):                                           # the second step shows the steady state
    t = time.time()
    model(x).logsumexp(1).sum().backward()
    print(f"step {i}: {time.time() - t:.1f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
print(f"{size}x{size} L={layers} batch={batch} {memory} segment={segment} | torch {torch.__version__} pennylane {pennylane.__version__} "
      f"numpy {numpy.__version__} python {sys.version.split()[0]}")
