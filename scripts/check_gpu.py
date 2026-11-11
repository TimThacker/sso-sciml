"""Report the GPU and time the two workloads the project plan leans on.

1. Batched non-symmetric eigenvalue solves (small-signal data generation).
2. Mixed-precision MLP training steps (surrogate training).

Run with:  uv run scripts/check_gpu.py
"""

from __future__ import annotations

import platform
import time

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_eigvals(device: str, batch: int, n: int, repeats: int = 3) -> float:
    """Median seconds for one batched eigvals call on `batch` real n-by-n matrices."""
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(batch, n, n, dtype=torch.float64, generator=gen).to(device)
    torch.linalg.eigvals(a[:8])  # warm-up
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        torch.linalg.eigvals(a)
        sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def time_training(device: str, amp: bool, steps: int = 200) -> float:
    """Samples per second for a 4x512 MLP, batch 8192, AdamW."""
    torch.manual_seed(0)
    batch, n_in, n_out = 8192, 16, 8
    model = torch.nn.Sequential(
        torch.nn.Linear(n_in, 512), torch.nn.GELU(),
        torch.nn.Linear(512, 512), torch.nn.GELU(),
        torch.nn.Linear(512, 512), torch.nn.GELU(),
        torch.nn.Linear(512, n_out),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    x = torch.randn(batch, n_in, device=device)
    y = torch.randn(batch, n_out, device=device)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            loss = torch.nn.functional.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(10):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    return steps * batch / (time.perf_counter() - t0)


def main() -> None:
    print(f"python   {platform.python_version()}  ({platform.system()} {platform.release()})")
    print(f"torch    {torch.__version__}")
    print(f"cuda     available={torch.cuda.is_available()}  build={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to PyTorch.")

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"device   {props.name}  capability={props.major}.{props.minor}")
    print(f"memory   {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB")
    print(f"threads  {torch.get_num_threads()} CPU threads")

    print("\nbatched eigvals, float64 (seconds per call)")
    print(f"{'batch':>8} {'n':>4} {'cpu':>9} {'cuda':>9} {'cpu/cuda':>9}")
    for batch, n in [(4096, 20), (4096, 40), (1024, 100)]:
        t_cpu = time_eigvals("cpu", batch, n)
        t_gpu = time_eigvals("cuda", batch, n)
        print(f"{batch:>8} {n:>4} {t_cpu:>9.3f} {t_gpu:>9.3f} {t_cpu / t_gpu:>9.2f}")

    print("\nMLP training throughput (samples per second)")
    cpu = time_training("cpu", amp=False, steps=20)
    gpu32 = time_training("cuda", amp=False)
    gpu_amp = time_training("cuda", amp=True)
    print(f"  cpu  float32   {cpu:>12,.0f}")
    print(f"  cuda float32   {gpu32:>12,.0f}")
    print(f"  cuda AMP       {gpu_amp:>12,.0f}")
    print(f"\npeak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
