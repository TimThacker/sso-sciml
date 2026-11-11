# sso-sciml

Scientific machine learning for sub-synchronous oscillation (SSO) screening in series-compensated and inverter-rich power grids.

The question it keeps asking: will this machine, at this operating point, on this network, see negative slip at the electrical resonance and feed an oscillation, and how fast will it grow?

The project pairs a first-principles reference solver (network electromagnetic transients kept) with hand calculations of the mechanism and with several kinds of learned model: gradient-boosted trees, Gaussian processes, MLP ensembles, physics-informed networks, DeepONet, Fourier neural operators, neural ODEs and sequence models. Its central experiment compares where the physics goes: in the loss, in the input features, in the architecture, or nowhere.

**Status:** Phases 0 and 1 complete. The solver core is validated on the IEEE First Benchmark Model. No learned models yet.

| Document | What it is |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | Scope, physics, learning tasks, phases, exit checks, risks |
| [docs/steps/](docs/steps/) | Step log: what was run, the numbers, what disagreed |
| [docs/EQUATIONS.md](docs/EQUATIONS.md) | Model equations and sign conventions |
| [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Each modelling assumption, how it was tested, and the result |

## Phase 1 in one figure

![Slip mechanism on the First Benchmark Model](docs/results/phase1/fig1_slip_mechanism.png)

On the benchmark, the rotor always sees negative slip at the electrical resonance and always presents negative resistance, but never enough to cancel the network's 0.03 pu. The hand estimate of the decay rate matches the full model within 2.6 %. What destabilises this system is the shaft: at the benchmark compensation the second torsional mode grows at +0.36 1/s. Details in [docs/steps/01-solver-core-fbm.md](docs/steps/01-solver-core-fbm.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for training, an NVIDIA GPU with a current driver. Developed on Windows 11 with an RTX 3070 (8 GB).

```powershell
git clone git@github.com:TimThacker/sso-sciml.git
cd sso-sciml
uv sync
uv run scripts/check_gpu.py
uv run pytest
```

`uv sync` installs the CUDA 13.0 build of PyTorch on Windows and the CPU build elsewhere.

## Everyday commands

| Task | Command |
|---|---|
| Sync environment to lockfile | `uv sync` |
| Run tests | `uv run pytest` |
| Run tests without a GPU | `uv run pytest -m "not gpu"` |
| Lint | `uv run ruff check` |
| Add a dependency | `uv add <package>` |
| GPU report and benchmark | `uv run scripts/check_gpu.py` |
| Reproduce Phase 1 numbers and figures | `uv run scripts/phase1_validate.py` |

## Licence

MIT. See [LICENSE](LICENSE).
