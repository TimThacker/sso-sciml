"""The admittance of a converter along the power axis.

Step 26 fixed the power and moved the other number of the operating point, u (reactive
current with a phase-locked loop, terminal voltage without). Here both move. The
admittance at the operating point of a case, (P, u), is built from scans taken elsewhere:

    rated            one scan at rated power and the reference u (zero reactive current,
                     or 1 pu voltage): the usual single scan
    rated, right u   rated power, the u of the case
    right P, ref u   the power of the case, the reference u
    2 x 2            scans at 0.1 and 1 pu of power and at the ends of the range of u;
                     straight lines both ways
    3 x 3            scans at 0.1, 0.55 and 1 pu and at three values of u; parabolas
    3 in power       scans at 0.1, 0.55 and 1 pu at the u of the case; a parabola
    exact            the scan at (P, u)

Power keeps its sign: a source is built from source scans, a load from load scans.

Used by scripts/power_axis.py.
"""

from __future__ import annotations

import numpy as np

from . import admittance as adm
from .data import c1
from .physics.converter import Converter

KINDS = ("pll", "follow")
POWER = np.array([0.1, 0.55, 1.0])  # size of the power at which scans are taken, pu
U_GRID = {"pll": np.array([-0.5, -0.15, 0.2]), "follow": np.array([0.7, 0.85, 1.0])}
U_REF = {"pll": 0.0, "follow": 1.0}
METHODS = ("rated", "rated, right u", "right P, ref u", "2 x 2", "3 x 3", "3 in power", "exact")
SCANS = {"rated": 1, "rated, right u": 1, "right P, ref u": 1, "2 x 2": 4, "3 x 3": 9,
         "3 in power": 3, "exact": 1}  # fmt: skip
COLUMNS = ("p", "u", "true", *METHODS)
_GRID = adm.scan_grid()[:: adm.NOISE_STEP]
_S = 2j * np.pi * _GRID


def scan(model: Converter, p_src: float, u: float) -> np.ndarray:
    """Admittance of the converter alone at power p_src and operating point u, on the
    251-point grid, with the terminal voltage on the real axis."""
    if model.controls.follow:
        v_t = complex(u, 0.0)
        i, p = model.current_at(v_t, p_src), model.params(p_src)
    else:
        v_t = 1.0 + 0.0j
        i, p = model.current_at(v_t, p_src, u), model.params(p_src, iq_ref=u)
    return adm.ConverterScan(model, v_t, i, p).admittance(_S)


def weights(x: float, nodes: np.ndarray) -> np.ndarray:
    """Lagrange weights of the polynomial through `nodes`, evaluated at x."""
    w = np.ones(len(nodes))
    for j, a in enumerate(nodes):
        for k, b in enumerate(nodes):
            if k != j:
                w[j] *= (x - b) / (a - b)
    return w


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS. NaN without a steady state."""
    kind, theta = args
    follow = float(kind == "follow")
    try:
        model, x0, p, net = c1.operating_point(theta, follow)
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return np.full(len(COLUMNS), np.nan)
    p0 = float(theta[c1.NAMES.index("p")])
    if follow:
        vt = model.terminal_voltage(x0, p)
        u0 = float(np.hypot(float(vt[0]), float(vt[1])))
    else:
        u0 = float(p["iq_ref"])
    sign = 1.0 if p0 >= 0 else -1.0
    powers, us, ref = sign * POWER, U_GRID[kind], U_REF[kind]
    wp3, wu3 = weights(p0, powers), weights(u0, us)
    wp2, wu2 = weights(p0, powers[[0, 2]]), weights(u0, us[[0, 2]])
    with np.errstate(all="ignore"):
        grid = [[scan(model, a, b) for b in us] for a in powers]
        column = [scan(model, a, u0) for a in powers]
        built = {
            "rated": scan(model, powers[2], ref),
            "rated, right u": column[2],
            "right P, ref u": scan(model, p0, ref),
            "2 x 2": sum(wp2[i] * wu2[j] * grid[2 * i][2 * j] for i in range(2) for j in range(2)),
            "3 x 3": sum(wp3[i] * wu3[j] * grid[i][j] for i in range(3) for j in range(3)),
            "3 in power": sum(wp3[i] * column[i] for i in range(3)),
            "exact": scan(model, p0, u0),
        }
    z = adm.network_impedance(net, _S)
    out = [p0, u0, c1.growth(model, x0, p)[0]]
    for name in METHODS:
        y = built[name]
        out.append(adm.fit_growth(y, z, _GRID)[0] if np.isfinite(y).all() else np.nan)
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
