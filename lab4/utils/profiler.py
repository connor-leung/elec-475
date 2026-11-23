import time
import torch
from .device import synchronize


@torch.no_grad()
def ms_per_image(model, device, size=320, runs=50):
    """Average inference time per 1-image batch, in milliseconds."""
    x = torch.randn(1, 3, size, size, device=device)
    model.eval()
    for _ in range(10):
        model(x)
    synchronize(device)
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    synchronize(device)
    return 1000 * (time.perf_counter() - t) / runs
