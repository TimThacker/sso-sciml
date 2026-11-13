"""Several converters on one bus: a source and a load side by side.

Each unit is the generic converter of step 17 with its own controls, power and share of
the common rating. All see the same terminal voltage. The network sees the sum of their
currents, so the admittance of the bus is the sum of theirs, weighted by share:

    Y = sum_k share_k Y_k

The modes of the whole are those of the units' state-space models side by side, closed
on the network exactly (`vectorfit.closed_loop_modes`).

One unit with a phase-locked loop may regulate: its reactive current is the one that
holds the bus voltage at 1 pu. The others run at zero reactive current, or follow the
voltage waveform if they have no phase-locked loop. With no regulating unit the voltage
is whatever the network gives.

Used by scripts/two_converters.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import fsolve

from . import admittance as adm
from . import vectorfit as vf
from .data import c1
from .physics.converter import Converter

LOADS = ("pll", "follow")  # kind of the load unit
COLUMNS = (
    "share",  # of the source in the common rating
    "p_source", "p_load",  # power of each on its own rating, pu; the load's is negative
    "p_net",  # power out of the bus on the common rating
    "solved", "together",  # largest growth rate of the two together, 1/s
    "solved_source", "source_alone",  # the source by itself on the same network
    "solved_load", "load_alone",
    "scans_added",  # each scanned by itself on the network, the scans added
    "solved_net", "net_equivalent",  # one converter carrying the net power
    "v_load_alone", "iq_together", "iq_source_alone",
)  # fmt: skip


@dataclass(frozen=True)
class Unit:
    model: Converter
    p_src: float  # power out of the unit on its own rating; negative for a load
    share: float  # of the common rating
    regulate: bool = False


def _z60(net: dict) -> complex:
    line = net["r_e"] + 1j * (net["x_e"] - net["x_c"])
    path = net["r_p"] + 1j * net["x_p"]
    return net["r_t"] + 1j * net["x_t"] + line * path / (line + path)


def operating_point(units: list[Unit], net: dict) -> tuple[complex, list[complex], list[float]]:
    """Bus voltage, the current out of each unit (on its own rating) and its reactive
    current reference. Raises ValueError if there is no steady state."""
    z_g = _z60(net)
    lead = next((k for k, u in enumerate(units) if u.regulate), None)
    if lead is not None and units[lead].model.controls.follow:
        raise ValueError("a unit without a phase-locked loop cannot regulate")

    def unpack(w):
        if lead is None:
            return complex(w[0], w[1]), 0.0
        return np.exp(1j * w[0]), float(w[1])

    def currents(v: complex, iq: float) -> tuple[list[complex], list[float]]:
        out, refs = [], []
        for k, u in enumerate(units):
            ref = iq if k == lead else 0.0
            if u.model.controls.follow:
                out.append(u.model.current_at(v, u.p_src))
            else:
                out.append(u.model.current_at(v, u.p_src, ref))
            refs.append(ref)
        return out, refs

    def mismatch(w):
        v, iq = unpack(w)
        cur, _ = currents(v, iq)
        r = v - 1.0 - z_g * sum(u.share * i for u, i in zip(units, cur, strict=True))
        return [r.real, r.imag]

    with np.errstate(all="ignore"):
        start = [1.0, 0.0] if lead is None else [0.0, 0.0]
        w = fsolve(mismatch, start, xtol=1e-12, full_output=True)[0]
        ok = np.all(np.isfinite(w)) and np.max(np.abs(mismatch(w))) < 1e-9
    if not ok:
        raise ValueError("steady state not found")
    v, iq = unpack(w)
    cur, refs = currents(v, iq)
    return v, cur, refs


def plant(units: list[Unit], v: complex, cur: list[complex], refs: list[float]):
    """The units side by side as dx/dt = A x + B v, current drawn = C x."""
    scans = [
        adm.ConverterScan(u.model, v, i, u.model.params(u.p_src, iq_ref=ref))
        for u, i, ref in zip(units, cur, refs, strict=True)
    ]
    return scans


def side_by_side(scans: list[adm.Scan], shares: list[float]):
    a = block_diag(*[s.a for s in scans])
    b = np.concatenate([s.b for s in scans])
    c = np.concatenate([w * s.c for s, w in zip(scans, shares, strict=True)], axis=1)
    return a, b, c


def growth(scans: list[adm.Scan], shares: list[float], net: dict) -> float:
    """Largest growth rate (1/s) of the units together on the network."""
    a, b, c = side_by_side(scans, shares)
    return float(vf.closed_loop_modes(a, b, c, np.zeros((2, 2)), net).real.max())


def _solve(units: list[Unit], net: dict):
    """Scans at the steady state of these units on the network, with the bus voltage
    and the regulating unit's reactive current. None without a steady state."""
    try:
        v, cur, refs = operating_point(units, net)
        return plant(units, v, cur, refs), abs(v), max(refs, key=abs)
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return None


def case(args: tuple[str, np.ndarray, np.ndarray, float]) -> np.ndarray:
    """One case in the order of COLUMNS. `theta_s` and `theta_l` are rows of dataset
    C1's parameters: the source takes its controls, the size of its power and the network
    from the first; the load takes controls and the size of its power from the second."""
    load, theta_s, theta_l, share = args
    ths = dict(zip(c1.NAMES, map(float, theta_s), strict=True))
    thl = dict(zip(c1.NAMES, map(float, theta_l), strict=True))
    net = c1.network(ths)
    follow = float(load == "follow")
    p_s, p_l = abs(ths["p"]), -abs(thl["p"])
    src = Unit(Converter(c1.controls(ths, 0.0)), p_s, share, regulate=True)
    lod = Unit(Converter(c1.controls(thl, follow)), p_l, 1.0 - share, regulate=False)
    lod_alone = Unit(lod.model, p_l, 1.0 - share, regulate=not follow)
    p_net = share * p_s + (1.0 - share) * p_l
    # One converter with the net power: the controls of whichever unit dominates.
    lead, kind = (ths, 0.0) if p_net >= 0 else (thl, follow)
    net_unit = Unit(Converter(c1.controls(lead, kind)), p_net, 1.0, regulate=not kind)

    out = np.full(len(COLUMNS), np.nan)
    out[:4] = [share, p_s, p_l, p_net]
    both, s_alone, l_alone, one = (_solve(u, net) for u in ([src, lod], [src], [lod_alone],
                                                            [net_unit]))  # fmt: skip
    out[4], out[6], out[8], out[11] = (float(x is not None) for x in (both, s_alone, l_alone, one))
    if both:
        out[5], out[14] = growth(both[0], [share, 1.0 - share], net), both[2]
    if s_alone:
        out[7], out[15] = growth(s_alone[0], [share], net), s_alone[2]
    if l_alone:
        out[9], out[13] = growth(l_alone[0], [1.0 - share], net), l_alone[1]
    if s_alone and l_alone:
        out[10] = growth([s_alone[0][0], l_alone[0][0]], [share, 1.0 - share], net)
    if one:
        out[12] = growth(one[0], [1.0], net)
    return out


def sample(n: int, load: str) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """n cases: two independent rows of dataset C1's parameter space and a share between
    0.2 and 0.8. The same (n, load) always returns the same cases."""
    theta = c1.sample(2 * n, "pll", "in_distribution")
    rng = np.random.default_rng(c1.SEED + 27 + LOADS.index(load))
    order = rng.permutation(2 * n)
    theta_s, theta_l = theta[order[:n]], theta[order[n:]]
    shares = rng.uniform(0.2, 0.8, size=n)
    return [(load, s, lo, float(a)) for s, lo, a in zip(theta_s, theta_l, shares, strict=True)]


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def cases(jobs: list, processes: int = 1) -> np.ndarray:
    """`case` for every job. Order is kept."""
    if processes <= 1:
        return np.array([case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, jobs, chunksize=32))
