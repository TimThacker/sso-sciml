"""Screens given network data with error.

The plant is known exactly. The screen is handed a network whose five quantities are
each wrong by an independent factor:

    line reactance, line resistance, series capacitor, parallel path (R and X together),
    plant-side impedance (R and X together)

Three screens are evaluated on the wrong network and scored against the modes of the
full model on the right one:

    exact     the plant model closed on the network (what a clean scan and fit give)
    root      the hand formula of step 14, doubly-fed plant only
    phasor    the slow mode of the phasor-domain model, doubly-fed plant only, with the
              operating point also taken from the wrong network

Used by scripts/network_error.py.
"""

from __future__ import annotations

import numpy as np

from . import admittance as adm
from . import vectorfit as vf
from .data import c1, t1
from .physics.converter import active_states
from .scanfit import BANDS, band_growth
from .screening import PHASOR_FAST
from .solver.linearize import jacobian, quasi_static

KINDS = ("t1", "pll", "follow")
FACTORS = ("line reactance", "line resistance", "series capacitor", "parallel path",
           "plant-side impedance")  # fmt: skip
LEVELS = (0.02, 0.05, 0.10, 0.20)  # standard deviation of the log of each factor
SINGLE = 1.10  # one quantity at a time, this factor up and down
COLUMNS = (
    *(f"true_{b}" for b in BANDS),
    *(f"{name}_e{e:g}" for e in LEVELS
      for name in (*(f"exact_{b}" for b in BANDS), "root", "root_f", "phasor_slow")),
    *(f"{band}_{j}_{d}" for j in range(len(FACTORS)) for d in ("up", "down")
      for band in ("sub", "slow")),
)  # fmt: skip


def perturb(net: dict, factor: np.ndarray) -> dict:
    """The network with each of the five quantities multiplied by its factor."""
    return {
        "x_e": net["x_e"] * factor[0], "r_e": net["r_e"] * factor[1],
        "x_c": net["x_c"] * factor[2],
        "x_p": net["x_p"] * factor[3], "r_p": net["r_p"] * factor[3],
        "x_t": net["x_t"] * factor[4], "r_t": net["r_t"] * factor[4],
    }  # fmt: skip


def _plant(kind: str, theta: np.ndarray):
    """Modes of the full model, the plant alone as (a, b, c) and the network."""
    if kind == "t1":
        model, x0, p = t1.operating_point(theta)
        lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())
        return lam, adm.DFIGScan(model, x0, p), t1.network(theta)
    follow = float(kind == "follow")
    model, x0, p, net = c1.operating_point(theta, follow)
    keep = active_states(follow)
    lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy()[np.ix_(keep, keep)])
    vt = model.terminal_voltage(x0, p)
    i = complex(float(x0[0] + x0[2]), float(x0[1] + x0[3]))
    return lam, adm.ConverterScan(model, complex(float(vt[0]), float(vt[1])), i, p), net


def _phasor_slow(theta: np.ndarray, net: dict) -> float:
    try:
        model, x0, p = t1.operating_point(theta, net)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return float("nan")
    a = jacobian(model.rhs, x0, p).numpy()
    return band_growth(np.linalg.eigvals(quasi_static(a, PHASOR_FAST)[0]))[0]


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS. NaN without a steady state."""
    kind, theta = args
    try:
        lam, plant, net = _plant(kind, theta)
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
            np.linalg.LinAlgError):  # fmt: skip
        return np.full(len(COLUMNS), np.nan)
    zero = np.zeros((2, 2))

    def exact(other: dict) -> list[float]:
        return band_growth(vf.closed_loop_modes(plant.a, plant.b, plant.c, zero, other))

    rng = np.random.default_rng(t1.case_seed(theta) + 3)
    out = band_growth(lam)
    for level in LEVELS:
        wrong = perturb(net, np.exp(level * rng.normal(size=len(FACTORS))))
        out += exact(wrong)
        if kind == "t1":
            out += [*t1.impedance_root(theta, wrong), _phasor_slow(theta, wrong)]
        else:
            out += [np.nan, np.nan, np.nan]
    for j in range(len(FACTORS)):
        for value in (SINGLE, 1.0 / SINGLE):
            factor = np.ones(len(FACTORS))
            factor[j] = value
            slow, sub, _ = exact(perturb(net, factor))
            out += [sub, slow]
    return np.array(out, dtype=float)


BOUNDS = (0.05, 0.10, 0.20)  # each network quantity known to within this, either way
# The end of each range that pushes the doubly-fed plant toward instability (step 22):
# less line reactance, less line resistance, more capacitor, a weaker parallel path,
# less plant-side impedance.
UNFAVOURABLE = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
CORNER_COLUMNS = (
    *(f"true_{b}" for b in BANDS),
    *(f"{name}_b{e:g}" for e in BOUNDS
      for name in (*(f"nominal_{b}" for b in BANDS), *(f"corners_{b}" for b in BANDS),
                   *(f"known_{b}" for b in BANDS), "root_nominal", "root_f_nominal",
                   "root_known", "root_f_known")),
)  # fmt: skip


def corner_case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of CORNER_COLUMNS: the screen is given a network that is
    wrong by up to a bound in each quantity, and it knows the bound.

    nominal   the plant closed on the network as given
    corners   the same on all 32 ends of the ranges; the largest growth rate in each band
    known     on the one end that is unfavourable for the doubly-fed plant
    root      the hand formula of step 14, as given and at that end (doubly-fed only)
    """
    kind, theta = args
    try:
        lam, plant, net = _plant(kind, theta)
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
            np.linalg.LinAlgError):  # fmt: skip
        return np.full(len(CORNER_COLUMNS), np.nan)
    zero = np.zeros((2, 2))

    def exact(other: dict) -> list[float]:
        return band_growth(vf.closed_loop_modes(plant.a, plant.b, plant.c, zero, other))

    rng = np.random.default_rng(t1.case_seed(theta) + 4)
    signs = np.array(np.meshgrid(*[(-1.0, 1.0)] * len(FACTORS))).reshape(len(FACTORS), -1).T
    out = band_growth(lam)
    for bound in BOUNDS:
        width = np.log1p(bound)
        given = perturb(net, np.exp(rng.uniform(-width, width, size=len(FACTORS))))
        worst = np.max([exact(perturb(given, np.exp(width * s))) for s in signs], axis=0)
        end = perturb(given, np.exp(width * UNFAVOURABLE))
        out += [*exact(given), *worst, *exact(end)]
        if kind == "t1":
            out += [*t1.impedance_root(theta, given), *t1.impedance_root(theta, end)]
        else:
            out += [np.nan] * 4
    return np.array(out, dtype=float)


def corner_cases(kind: str, theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`corner_case` for every row of theta. Row order is kept."""
    jobs = [(kind, row) for row in theta]
    if processes <= 1:
        return np.array([corner_case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(corner_case, jobs, chunksize=64))


# (noise on each entry of the scan, bound on each network quantity)
BOTH = ((0.0, 0.10), (0.01, 0.0), (0.01, 0.05), (0.01, 0.10), (0.05, 0.10))
BOTH_COLUMNS = (
    *(f"true_{b}" for b in BANDS),
    *(f"{name}_n{e:g}_b{w:g}" for e, w in BOTH
      for name in (*(f"nominal_{b}" for b in BANDS), *(f"corners_{b}" for b in BANDS))),
)  # fmt: skip
_GRID = adm.scan_grid()
_HALF = _GRID >= 0


def both_case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of BOTH_COLUMNS: a noisy 126-point scan of the plant,
    fitted by least squares (step 20), closed on a network that is wrong by up to a
    bound in each quantity, as given and at the 32 ends of the ranges (step 23)."""
    from .scanfit import _problem, noisy

    kind, theta = args
    try:
        lam, y, _, net = _problem(kind, theta)
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
            np.linalg.LinAlgError):  # fmt: skip
        return np.full(len(BOTH_COLUMNS), np.nan)
    rng = np.random.default_rng(t1.case_seed(theta) + 5)
    signs = np.array(np.meshgrid(*[(-1.0, 1.0)] * len(FACTORS))).reshape(len(FACTORS), -1).T
    f, y_half = _GRID[_HALF][::16], y[_HALF][::16]
    out = band_growth(lam)
    for level, bound in BOTH:
        fit = vf.fit_admittance(f, noisy(y_half, level, rng), level)

        def modes(other: dict, fit=fit) -> list[float]:
            return band_growth(vf.closed_loop_modes(fit["a"], fit["b"], fit["c"], fit["d"], other))

        width = np.log1p(bound)
        given = perturb(net, np.exp(rng.uniform(-width, width, size=len(FACTORS))))
        nominal = modes(given)
        if bound == 0:
            out += [*nominal, *nominal]
            continue
        worst = np.max([modes(perturb(given, np.exp(width * s))) for s in signs], axis=0)
        out += [*nominal, *worst]
    return np.array(out, dtype=float)


def both_cases(kind: str, theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`both_case` for every row of theta. Row order is kept."""
    jobs = [(kind, row) for row in theta]
    if processes <= 1:
        return np.array([both_case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(both_case, jobs, chunksize=32))


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
