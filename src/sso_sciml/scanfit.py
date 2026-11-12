"""One case of either plant type as a noisy scan, fitted, with its modes sorted into
frequency bands and set beside those of the full model.

    slow   below 2 Hz in the synchronous frame: outer loops, within 2 Hz of 60 Hz in the phases
    sub    2 to 58 Hz: the sub-synchronous band and its mirror image
    fast   above 58 Hz

Used by scripts/phase3_noise_fits.py.
"""

from __future__ import annotations

import numpy as np

from . import admittance as adm
from . import vectorfit as vf
from .data import c1, t1
from .physics.converter import active_states
from .solver.linearize import jacobian

KINDS = ("t1", "pll", "follow")
BANDS = {"slow": (0.0, 2.0), "sub": (2.0, 58.0), "fast": (58.0, 2000.0)}
# (keep every n-th point of the scan, noise on each entry of Y) for the least-squares fit.
VARIANTS = ((16, 0.0), (16, 0.001), (16, 0.01), (16, 0.05), (4, 0.05))
AAA_NOISE = (1e-6, 1e-4, 1e-3, 1e-2)  # for the interpolating fit, on the 251-point scan
COLUMNS = (
    *(f"true_{b}" for b in BANDS),
    *(f"{name}_s{k}_n{e:g}" for k, e in VARIANTS
      for name in (*BANDS, "pairs", "states", "residual")),
    *(f"aaa_n{e:g}" for e in AAA_NOISE),
)  # fmt: skip
_GRID = adm.scan_grid()
_HALF = _GRID >= 0


def band_growth(lam: np.ndarray) -> list[float]:
    """Largest real part (1/s) of the modes in each band; -inf for an empty band."""
    f = np.abs(lam.imag) / (2 * np.pi)
    out = []
    for lo, hi in BANDS.values():
        inside = (f >= lo) & (f < hi) & (np.abs(lam.real) < 2 * np.pi * 2000.0)
        out.append(float(lam.real[inside].max()) if inside.any() else float("-inf"))
    return out


def noisy(y: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Each entry at each frequency multiplied by 1 + e, e complex normal."""
    if level == 0:
        return y
    err = rng.normal(size=(*y.shape, 2)) @ np.array([1.0, 1j]) * level / np.sqrt(2)
    return y * (1 + err)


def _problem(kind: str, theta: np.ndarray):
    """Modes of the full model, the scans and the network of one case."""
    if kind == "t1":
        lam = np.linalg.eigvals(t1.state_matrix(theta))
        y, z, _ = t1.scan_pair(theta)
        return lam, y, z, t1.network(theta)
    follow = float(kind == "follow")
    model, x0, p, net = c1.operating_point(theta, follow)
    keep = active_states(follow)
    lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy()[np.ix_(keep, keep)])
    y, z, _ = c1.scan_pair(theta, follow)
    return lam, y, z, net


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS. NaN without a steady state."""
    kind, theta = args
    try:
        lam, y, z, net = _problem(kind, theta)
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
            np.linalg.LinAlgError):  # fmt: skip
        return np.full(len(COLUMNS), np.nan)
    rng = np.random.default_rng(t1.case_seed(theta) + 2)
    out = band_growth(lam)
    for step, level in VARIANTS:
        f, y_scan = _GRID[_HALF][::step], noisy(y[_HALF][::step], level, rng)
        fit = vf.fit_admittance(f, y_scan, level)
        modes = vf.closed_loop_modes(fit["a"], fit["b"], fit["c"], fit["d"], net)
        out += [*band_growth(modes), fit["pairs"], fit["states"], fit["residual"]]
    step = adm.NOISE_STEP
    for level in AAA_NOISE:
        out.append(adm.fit_growth(noisy(y[::step], level, rng), z[::step], _GRID[::step],
                                  rtol=level)[0])  # fmt: skip
    return np.array(out, dtype=float)


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def cases(kind: str, theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`case` for every row of theta. Row order is kept."""
    jobs = [(kind, row) for row in theta]
    if processes <= 1:
        return np.array([case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, jobs, chunksize=32))
