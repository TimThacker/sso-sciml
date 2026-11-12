"""The slow mode of the doubly-fed plant (System B2, full controls), below 2 Hz in the
synchronous frame, and the reduced models that are asked to reproduce it.

    full       25 states
    phasor     17 states: network and stator transients made algebraic (step 06)
    loops       9 states: rotor flux, grid-side converter and DC link algebraic as well.
                What is left is rotor speed, the phase-locked loop, the rotor current
                controller integrators, the power filters and the outer-loop integrators
    held        8 states: the same with rotor speed held
    alone      the plant cut from its network, on an ideal voltage (21 states)

Used by scripts/slow_mode.py.
"""

from __future__ import annotations

import numpy as np

from . import admittance as adm
from .data import t1
from .scanfit import BANDS, band_growth
from .screening import PHASOR_FAST
from .solver.linearize import jacobian, modes, quasi_static

N_STATE = 25
SPEED = 8
# Fast states of the 9-state model: the phasor set, rotor flux, grid-side current
# integrators, DC link and its controller, feed-forward filter.
LOOPS_FAST = [*PHASOR_FAST, 2, 3, 17, 18, 19, 20, 21, 22]
COLUMNS = (
    *(f"true_{b}" for b in BANDS),
    "f_slow",  # frequency of the least-damped slow mode, Hz
    *(f"phasor_{b}" for b in BANDS),
    "loops_slow", "held_slow", "alone_slow",
    "half_outer_slow",  # full model, outer loops at half their speed
    "double_rsc_slow",  # full model, rotor current loop at twice its bandwidth
    *(f"part_{k}" for k in range(N_STATE)),  # participation of the states in that mode
)  # fmt: skip


def _slow(lam: np.ndarray) -> float:
    return band_growth(lam)[0]


def _changed(theta: np.ndarray, name: str, factor: float) -> float:
    other = theta.copy()
    other[t1.NAMES.index(name)] *= factor
    a = t1.state_matrix(other)
    return _slow(np.linalg.eigvals(a)) if np.isfinite(a).all() else float("nan")


def case(theta: np.ndarray) -> np.ndarray:
    """One case in the order of COLUMNS. NaN without a steady state."""
    try:
        model, x0, p = t1.operating_point(theta)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.full(len(COLUMNS), np.nan)
    a = jacobian(model.rhs, x0, p).numpy()
    md = modes(a)
    f = np.abs(md.freq_hz)
    lo, hi = BANDS["slow"]
    inside = np.where((f >= lo) & (f < hi))[0]
    top = inside[np.argmax(md.sigma[inside])]
    phasor, _ = quasi_static(a, PHASOR_FAST)
    loops, kept = quasi_static(a, LOOPS_FAST)
    rest = [j for j, k in enumerate(kept) if k != SPEED]
    plant = adm.DFIGScan(model, x0, p)
    return np.array(
        [
            *band_growth(md.eigenvalues), f[top],
            *band_growth(np.linalg.eigvals(phasor)),
            _slow(np.linalg.eigvals(loops)),
            _slow(np.linalg.eigvals(loops[np.ix_(rest, rest)])),
            _slow(plant.poles),
            _changed(theta, "outer_ratio", 0.5),
            _changed(theta, "rsc_hz", 2.0),
            *md.participation[:, top],
        ]
    )  # fmt: skip


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def cases(theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`case` for every row of theta. Row order is kept."""
    if processes <= 1:
        return np.array([case(row) for row in theta])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, list(theta), chunksize=64))
