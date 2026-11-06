# ELEC 475 Lab 3: Apple Silicon (MPS) Support

The starter scripts only chose between CUDA and the CPU. We added
`utils/device.py` so the code can also run on the GPU of Apple Silicon (M-series)
Macs through PyTorch's MPS backend. All results in the report were produced on an
Apple M4 Pro GPU this way.

## Device selection

`get_device()` picks the first available backend in this order:

1. **CUDA** (NVIDIA GPU)
2. **MPS** (Apple Silicon GPU)
3. **CPU**

CUDA is checked first, so on an NVIDIA or CPU-only machine the scripts behave
exactly like the starter code.

## Changes to the provided files

| File | Change |
|------|--------|
| `utils/device.py` | **New.** `get_device()`, `device_name()` (prints the GPU or chip name) and `synchronize()` (waits for CUDA or MPS work to finish) |
| `predict.py`, `train_correction.py`, `correct.py` | Use `get_device()` / `device_name()` instead of the CUDA-or-CPU check |
| `utils/profiler.py` | Uses `synchronize(device)` so inference timings are also accurate on MPS |
| `predict.py`, `train_correction.py` | DataLoader `num_workers` set to 4 (all DataLoader code is under `if __name__ == "__main__":`, so this is safe on Windows) |

No model, dataset, metric or evaluation logic was changed for this. To run
everything, see `commands.txt`.
