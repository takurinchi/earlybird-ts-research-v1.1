"""Synchronized inference latency measurements."""
import time
import torch


@torch.inference_mode()
def measure_inference(model, inputs, device, warmup=10, repeats=100):
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats >= 1")
    model.eval()
    inputs = inputs.to(device)
    def sync():
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        model(inputs)
    sync()
    samples = []
    for _ in range(repeats):
        sync(); start = time.perf_counter(); model(inputs); sync()
        samples.append(time.perf_counter() - start)
    return {"mean_seconds": sum(samples) / repeats,
            "median_seconds": sorted(samples)[repeats // 2],
            "repeats": repeats, "warmup": warmup}
