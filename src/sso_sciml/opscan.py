"""The admittance of a converter as a function of its operating point.

At a given power the admittance of the generic converter depends on one more number:

    with a phase-locked loop   the reactive current, since the terminal voltage is held
    without one                the terminal voltage, which the network sets

Call it u. An outage moves u, and step 25 showed that a scan taken at the old u clears
insecure cases. Here the scan after an outage is built from scans at a few values of u:

    reuse    the scan at the u before the outage
    two      straight line through scans at u0 and u0 - STEP
    three    parabola through scans at u0 - STEP, u0, u0 + STEP
    lib3     straight lines between 3 scans that span the whole range of u
    lib5     the same with 5
    exact    the scan at the u after the outage

Each is closed on the network after the outage by a rational fit to the determinant
(step 19; the scans are clean) and compared with the modes of the full model.

Used by scripts/operating_point.py.
"""

from __future__ import annotations

import numpy as np

from . import admittance as adm
from .data import c1
from .neterror import perturb
from .outage import OUTAGES
from .physics.converter import active_states
from .physics.induction import _t
from .solver.linearize import jacobian

KINDS = ("pll", "follow")
STEP = 0.1
RANGE = {"pll": (-0.8, 0.2), "follow": (0.65, 1.0)}  # of u, from the outages of step 25
METHODS = ("reuse", "two", "three", "lib3", "lib5", "exact")
COLUMNS = (
    "intact", "u0",
    *(f"{name}_{j}" for j in range(len(OUTAGES)) for name in ("solved", "true", "u1", *METHODS)),
)  # fmt: skip
_GRID = adm.scan_grid()[:: adm.NOISE_STEP]
_S = 2j * np.pi * _GRID


def scan_at(model, p: dict, p_src: float, u: float, follow: float) -> np.ndarray:
    """Admittance of the converter alone at operating point u, on the 251-point grid."""
    if follow:
        v_t = complex(u, 0.0)
        return adm.ConverterScan(model, v_t, model.current_at(v_t, p_src), p).admittance(_S)
    v_t = 1.0 + 0.0j
    i = model.current_at(v_t, p_src, u)
    return adm.ConverterScan(model, v_t, i, {**p, "iq_ref": _t(u)}).admittance(_S)


def _u(model, x0, p: dict, follow: float) -> float:
    if not follow:
        return float(p["iq_ref"])
    vt = model.terminal_voltage(x0, p)
    return float(np.hypot(float(vt[0]), float(vt[1])))


def _growth(model, x0, p: dict, follow: float) -> float:
    keep = active_states(follow)
    return float(
        np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy()[np.ix_(keep, keep)]).real.max()
    )


def _between(grid: np.ndarray, scans: list[np.ndarray], u: float) -> np.ndarray:
    """Straight lines between neighbouring scans; the end scan outside the range."""
    u = float(np.clip(u, grid[0], grid[-1]))
    k = int(np.clip(np.searchsorted(grid, u) - 1, 0, len(grid) - 2))
    w = (u - grid[k]) / (grid[k + 1] - grid[k])
    return (1 - w) * scans[k] + w * scans[k + 1]


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS. NaN without a steady state before the outage."""
    kind, theta = args
    follow = float(kind == "follow")
    failures = (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
                np.linalg.LinAlgError)  # fmt: skip
    try:
        model, x0, p, net = c1.operating_point(theta, follow)
    except failures:
        return np.full(len(COLUMNS), np.nan)
    p_src = float(theta[c1.NAMES.index("p")])
    u0 = _u(model, x0, p, follow)

    def scan(u: float) -> np.ndarray:
        return scan_at(model, p, p_src, u, follow)

    y0, y_lo, y_hi = scan(u0), scan(u0 - STEP), scan(u0 + STEP)
    grid5 = np.linspace(*RANGE[kind], 5)
    lib5 = [scan(u) for u in grid5]
    out = [_growth(model, x0, p, follow), u0]
    for factor in OUTAGES.values():
        other = perturb(net, np.array(factor))
        try:
            m2, x2, p2, _ = c1.operating_point(theta, follow, net=other)
        except failures:
            out += [0.0] + [np.nan] * (2 + len(METHODS))
            continue
        u1 = _u(m2, x2, p2, follow)
        d = (u1 - u0) / STEP
        built = {
            "reuse": y0,
            "two": y0 - d * (y_lo - y0),
            "three": y0 + 0.5 * d * (y_hi - y_lo) + 0.5 * d * d * (y_hi - 2 * y0 + y_lo),
            "lib3": _between(grid5[::2], lib5[::2], u1),
            "lib5": _between(grid5, lib5, u1),
            "exact": scan(u1),
        }
        z = adm.network_impedance(other, _S)
        out += [1.0, _growth(m2, x2, p2, follow), u1,
                *(adm.fit_growth(built[name], z, _GRID)[0] for name in METHODS)]  # fmt: skip
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
