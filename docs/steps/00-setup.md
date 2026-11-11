# 00: Project setup and GPU check

Date: 2026-11-11 · Plan phase: 0

## Goal

Create the project following the dev guide, confirm PyTorch trains on the local GPU, and measure the two workloads the plan depends on so that the compute split between CPU and GPU is decided from data.

## What was done

- `uv init --lib --python 3.12 sso-sciml` under `C:\Users\tntth\Code\dev`, branch renamed to `main`.
- `pyproject.toml`: PyTorch pulled from the CUDA 13.0 wheel index on Windows and from the CPU index elsewhere, so CI does not download CUDA libraries.
- `.gitignore` extended for datasets, checkpoints, and experiment tracking.
- `scripts/check_gpu.py`: device report plus two timings.
- `tests/test_environment.py`: import check, a known-spectrum eigenvalue check in float64, CPU against GPU agreement.
- `.github/workflows/ci.yml`: lint and CPU tests on every push.
- `docs/PLAN.md`.

## Commands

```powershell
cd C:\Users\tntth\Code\dev
uv init --lib --python 3.12 sso-sciml
cd sso-sciml
git branch -M main
uv sync
uv run scripts/check_gpu.py
uv run ruff check
uv run pytest
```

## Results

Output of `uv run scripts/check_gpu.py`:

```
python   3.12.13  (Windows 11)
torch    2.14.1+cu130
cuda     available=True  build=13.0
device   NVIDIA GeForce RTX 3070  capability=8.6
memory   6.95 GiB free of 8.00 GiB
threads  8 CPU threads

batched eigvals, float64 (seconds per call)
   batch    n       cpu      cuda  cpu/cuda
    4096   20     0.148     8.550      0.02
    4096   40     0.610    16.506      0.04
    1024  100     1.311    29.719      0.04

MLP training throughput (samples per second)
  cpu  float32        116,337
  cuda float32      2,406,917
  cuda AMP          3,104,608

peak GPU memory allocated: 160 MiB
```

- Training: the GPU with mixed precision is about 27 times faster than the CPU on a 4×512 MLP at batch 8192. Mixed precision adds about 29 % over float32 on the GPU.
- Eigenvalues: the CPU solves 4,096 matrices of size 20 in 0.15 s. The GPU takes 8.6 s for the same call.

## Disagreements

The expectation going in was that batched eigenvalue solves would be a GPU workload. They are not: the GPU is 23 to 58 times slower than the CPU across the three sizes tested. The non-symmetric eigenvalue routine does not run as a batched GPU kernel for small matrices, so the call pays per-matrix overhead. The plan was changed before it was written down: eigenvalue-based data generation is assigned to CPU processes.

Not yet tested: whether batched fixed-step time integration is faster on the GPU. That benchmark is in Phase 3.

## Decisions

- PyTorch is the single array framework. JAX has no GPU support on native Windows.
- Solver code in float64; surrogate training in float32 with mixed precision.
- Eigenvalue solves on the CPU. GPU for training, inference, and (pending the Phase 3 benchmark) batched time integration.
- Python pinned to 3.12.
- MIT licence.

## Next

Phase 1, step 1: write `docs/EQUATIONS.md` for the dq-frame network and the First Benchmark Model.
