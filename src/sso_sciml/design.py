"""The design problem of Phase 10: controller settings and compensation for damping.

Four quantities are chosen:

    rsc_hz      rotor current loop bandwidth, Hz, with a floor
    damp_gain   share of the loop's proportional gain the damping controller removes
    damp_hz     centre frequency of the damping controller, Hz
    segments    series-capacitor segments in service (integer); each compensates
                SEGMENT of the line reactance

The plant is `physics.dfig_damped.DFIGDamped` on the network of dataset T1. A design is
scored on a fixed set of scenarios (rotor speed, units online, strength of the parallel
path); its score is the largest growth rate, over the scenarios, of any oscillatory mode.
A design is a point of the unit cube: `decode` maps it to the four quantities.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import torch

from .data import t1
from .physics.dfig_damped import DFIGDamped
from .physics.dfig_full import Controls
from .physics.induction import IGData
from .solver.integrate import rk4
from .solver.linearize import jacobian, modes

SEGMENT = 0.15
MAX_SEGMENTS = 5
FLOOR_HZ = 10.0  # the control bandwidth floor
BOUNDS = {"rsc_hz": (FLOOR_HZ, 30.0), "damp_gain": (0.0, 0.9), "damp_hz": (5.0, 50.0)}
NAMES = ("rsc_hz", "damp_gain", "damp_hz", "segments")
# Everything the designer does not choose, at the reference values of dataset T1.
FIXED = {"pll_hz": 15.0, "outer_ratio": 0.07, "ff": 1.0, "x_line": 1.0, "xr_line": 23.3,
         "x_t": 0.15, "r_r": 0.00549}  # fmt: skip
SCENARIO_NAMES = ("w_r", "n_online", "par_ratio")
SCENARIOS = np.array(list(product((0.8, 1.0, 1.2), (0.4, 1.0), (1.0, 8.0))))
COLUMNS = ("sigma_sso", "f_sso", "sigma_osc", "f_osc", "sigma_max", "solved")
FAILED = 50.0  # score of a design with a scenario that has no steady state, 1/s
F_MIN = 2.0  # modes below this frequency are not counted as oscillations, Hz


def decode(z: np.ndarray) -> np.ndarray:
    """Points of the unit cube, (n, 4), to (rsc_hz, damp_gain, damp_hz, segments). The
    bandwidth is spaced in log. The last coordinate is cut into MAX_SEGMENTS equal parts."""
    z = np.clip(np.atleast_2d(z), 0.0, 1.0)
    lo, hi = np.log(BOUNDS["rsc_hz"][0]), np.log(BOUNDS["rsc_hz"][1])
    out = np.empty_like(z, dtype=float)
    out[:, 0] = np.exp(lo + z[:, 0] * (hi - lo))
    for j, key in ((1, "damp_gain"), (2, "damp_hz")):
        out[:, j] = BOUNDS[key][0] + z[:, j] * (BOUNDS[key][1] - BOUNDS[key][0])
    out[:, 3] = np.minimum(np.floor(z[:, 3] * MAX_SEGMENTS), MAX_SEGMENTS - 1) + 1
    return out


def theta_of(k: float, rsc_hz: float, scenario: np.ndarray) -> np.ndarray:
    """The 12 parameters of dataset T1 for a compensation, a bandwidth and a scenario."""
    th = {**FIXED, "k": k, "rsc_hz": rsc_hz, **dict(zip(SCENARIO_NAMES, scenario, strict=True))}
    return np.array([th[name] for name in t1.NAMES])


def operating_point(theta: np.ndarray, damp_gain: float, damp_hz: float):
    """The damped plant of one case with its steady state and parameters. Raises
    ValueError if the steady state is not found."""
    th = dict(zip(t1.NAMES, map(float, theta), strict=True))
    data = replace(IGData(), r_r=th["r_r"])
    net = t1._network(th, data)
    ctl = Controls(rsc_hz=th["rsc_hz"], pll_hz=th["pll_hz"], ff=th["ff"],
                   outer_hz=th["outer_ratio"] * th["rsc_hz"])  # fmt: skip
    model = DFIGDamped(data=data, controls=ctl, damp_gain=damp_gain, damp_hz=damp_hz)
    x0, p = model.equilibrium(net.pop("x_c"), w_r=th["w_r"], **net)
    if float(model.rhs(x0, p).abs().max()) > 1e-8:
        raise ValueError("steady state not found")
    return model, x0, p


def solve(case: np.ndarray) -> np.ndarray:
    """One solver call. `case` is the 12 parameters of T1 followed by damp_gain and
    damp_hz. Returns COLUMNS: the sub-synchronous mode (the one the series capacitor
    takes most part in), the least-damped oscillatory mode, and the largest real part."""
    try:
        model, x0, p = operating_point(case[:12], float(case[12]), float(case[13]))
        md = modes(jacobian(model.rhs, x0, p))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.0])
    cand = np.where((md.freq_hz > F_MIN) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    osc = np.where(md.freq_hz > F_MIN)[0]
    m = int(osc[np.argmax(md.sigma[osc])])
    return np.array([md.sigma[j], md.freq_hz[j], md.sigma[m], md.freq_hz[m], md.sigma.max(), 1.0])


_POOL: dict = {}


def close_pool() -> None:
    for pool in _POOL.values():
        pool.terminate()
    _POOL.clear()


def solve_many(cases: np.ndarray, processes: int = 1) -> np.ndarray:
    """`solve` for every row. Row order is kept. The worker processes are started once
    and kept, since a search calls this many times with a few rows; `close_pool` ends
    them."""
    if processes <= 1 or len(cases) < 2 * processes:
        return np.array([solve(row) for row in cases])
    if processes not in _POOL:
        from multiprocessing import Pool

        _POOL[processes] = Pool(processes, initializer=t1._init_worker)
    chunk = max(1, min(16, len(cases) // (2 * processes)))
    return np.array(_POOL[processes].map(solve, list(cases), chunksize=chunk))


def cases_of(designs: np.ndarray, scenarios: np.ndarray = SCENARIOS) -> np.ndarray:
    """Solver inputs for every design on every scenario, (designs * scenarios, 14),
    designs outermost."""
    rows = [np.r_[theta_of(SEGMENT * d[3], d[0], s), d[1], d[2]]
            for d in designs for s in scenarios]  # fmt: skip
    return np.array(rows)


def score(out: np.ndarray, n_designs: int) -> np.ndarray:
    """Worst growth rate of an oscillatory mode over the scenarios of each design."""
    sigma = np.where(out[:, 5] > 0, out[:, 2], FAILED).reshape(n_designs, -1)
    return sigma.max(axis=1)


def worst_case(designs: np.ndarray, processes: int = 1,
               scenarios: np.ndarray = SCENARIOS) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Score of each design, and the solver outputs behind it, (designs, scenarios, 6)."""
    designs = np.atleast_2d(designs)
    out = solve_many(cases_of(designs, scenarios), processes)
    return score(out, len(designs)), out.reshape(len(designs), len(scenarios), -1)


def time_response(design: np.ndarray, scenario: np.ndarray, offset: float = 0.05,
                  t_end: float = 1.0, dt: float = 2e-4) -> dict:  # fmt: skip
    """The nonlinear plant after the capacitor voltage is displaced by `offset` pu on
    the D axis. Returns the time, the deviation of the capacitor voltage, and the growth
    rate fitted to the size of the deviation over the second half of the run."""
    return time_responses(design, np.atleast_2d(scenario), offset, t_end, dt)[0]


def time_responses(design: np.ndarray, scenarios: np.ndarray = SCENARIOS, offset: float = 0.05,
                   t_end: float = 1.0, dt: float = 2e-4) -> list[dict]:  # fmt: skip
    """`time_response` for several scenarios, integrated together."""
    cases = cases_of(np.atleast_2d(design), scenarios)
    points = [operating_point(c[:12], float(c[12]), float(c[13])) for c in cases]
    model = points[0][0]
    x0 = torch.stack([x for _, x, _ in points])
    p = {k: torch.stack([q[k] for _, _, q in points]) for k in points[0][2]}
    start = x0.clone()
    start[:, 4] += offset
    t, x = rk4(model.rhs, start, p, t_end, dt, save_every=5)
    t = t.numpy()
    dev_all = (x[:, :, 4:6] - x0[None, :, 4:6]).numpy()
    edges = np.arange(0.5 * t_end, t_end + 1e-9, 0.05)
    out = []
    for k in range(len(cases)):
        dev = dev_all[:, k]
        size = np.sqrt((dev**2).sum(axis=1))
        finite = bool(np.isfinite(size).all())
        rate = float("nan")
        if finite:
            # Peaks of the size over windows of 50 ms, then a line through their logarithm.
            mid, top = [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                m = (t >= lo) & (t < hi)
                mid.append(0.5 * (lo + hi))
                top.append(size[m].max())
            if min(top) > 0:
                rate = float(np.polyfit(mid, np.log(top), 1)[0])
        out.append({"t": t, "deviation": dev[:, 0], "size": size, "rate": rate, "finite": finite,
                    "final_over_initial": float(size[-1] / offset) if finite
                    else float("inf")})  # fmt: skip
    return out
