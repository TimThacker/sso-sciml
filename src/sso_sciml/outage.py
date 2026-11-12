"""One outage at a time: a structured change to the network, not a small error in it.

For each case and each outage:

    true     the modes of the full model after the outage, at its new steady state
    reuse    the plant model taken at the steady state before the outage (one scan),
             closed on the network after it

`reuse` asks whether one scan of the plant can serve every network condition.

Used by scripts/outages.py.
"""

from __future__ import annotations

import numpy as np

from . import vectorfit as vf
from .data import c1, t1
from .neterror import _plant, perturb
from .physics.converter import active_states
from .scanfit import BANDS, band_growth
from .solver.linearize import jacobian

KINDS = ("t1", "pll", "follow")
# Factors on (line reactance, line resistance, series capacitor, parallel path,
# plant-side impedance). Impedances are on the base of the plant in service, so with
# half the units out every network impedance halves.
OUTAGES = {
    "parallel path out": (1.0, 1.0, 1.0, 100.0, 1.0),
    "one of two parallel circuits out": (1.0, 1.0, 1.0, 2.0, 1.0),
    "half the capacitor out": (1.0, 1.0, 0.5, 1.0, 1.0),
    "half the units out": (0.5, 0.5, 0.5, 0.5, 0.5),
}
COLUMNS = (
    *(f"intact_{b}" for b in BANDS),
    *(f"{name}_{j}" for j in range(len(OUTAGES))
      for name in ("solved", *(f"true_{b}" for b in BANDS), *(f"reuse_{b}" for b in BANDS))),
)  # fmt: skip


def _after(kind: str, theta: np.ndarray, net: dict) -> np.ndarray:
    """Modes of the full model on another network. Raises ValueError without a steady
    state."""
    if kind == "t1":
        model, x0, p = t1.operating_point(theta, net)
        return np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())
    follow = float(kind == "follow")
    model, x0, p, _ = c1.operating_point(theta, follow, net=net)
    keep = active_states(follow)
    return np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy()[np.ix_(keep, keep)])


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS. NaN without a steady state before the outage."""
    kind, theta = args
    failures = (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
                np.linalg.LinAlgError)  # fmt: skip
    try:
        lam, plant, net = _plant(kind, theta)
    except failures:
        return np.full(len(COLUMNS), np.nan)
    zero = np.zeros((2, 2))
    out = band_growth(lam)
    for factor in OUTAGES.values():
        other = perturb(net, np.array(factor))
        try:
            true = [1.0, *band_growth(_after(kind, theta, other))]
        except failures:
            true = [0.0, np.nan, np.nan, np.nan]
        reuse = band_growth(vf.closed_loop_modes(plant.a, plant.b, plant.c, zero, other))
        out += [*true, *reuse]
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
        return np.array(pool.map(case, jobs, chunksize=64))
