"""Dataset C1: a generic converter (System C) on a series-compensated or weak network.

One case is a point in a 12-dimensional parameter space and a family:

    pll     synchronised by a phase-locked loop; power out or in
    follow  no phase-locked loop, current follows the voltage waveform; power in only

For each case this module returns the modes of the full model and what each screen says
from the data it is allowed: the short-circuit ratio, scans of the converter admittance
and the network impedance, and the full model with one control loop slowed down.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np

from .. import admittance as adm
from ..physics.converter import Converter, ConverterControls, active_states
from ..physics.induction import _t
from ..solver.linearize import jacobian
from .t1 import Param, case_seed

VERSION = "c1-v1"
SEED = 20261112
X_T = 0.10  # transformer reactance, pu
XR_PLANT = 15.0
FAMILIES = ("pll", "follow")

SPEC: tuple[Param, ...] = (
    Param("p", -1.0, 1.0, False, "power out of the converter, pu; negative for a load",
          "rated power either way"),
    Param("scr", 1.5, 8.0, True, "short-circuit ratio at the terminal, compensation left out",
          "weak-grid practice counts below 3 as weak"),
    Param("k", 0.0, 0.80, False, "series compensation as a fraction of line reactance",
          "as in dataset T1, down to none"),
    Param("xr", 5.0, 30.0, True, "X/R of line and parallel path", "distribution to transmission"),
    Param("par_ratio", 0.5, 8.0, True, "parallel path impedance as a multiple of the line's",
          "as in dataset T1"),
    Param("cc_hz", 50.0, 500.0, True, "current loop bandwidth, Hz", "representative"),
    Param("pll_hz", 2.0, 50.0, True, "phase-locked loop natural frequency, Hz", "representative"),
    Param("dc_hz", 2.0, 30.0, True, "DC voltage loop natural frequency, Hz", "representative"),
    Param("ff", 0.0, 1.0, False, "terminal-voltage feed-forward gain", "off and on, as in T1"),
    Param("t_ff", 0.0005, 0.005, True, "voltage measurement filter, s", "representative"),
    Param("x_f", 0.10, 0.30, False, "filter reactance, pu", "representative"),
    Param("c_dc", 0.01, 0.05, True, "DC capacitance, pu s", "around the 0.02 of step 04"),
)  # fmt: skip
NAMES = tuple(p.name for p in SPEC)

SPLITS: dict[str, dict[str, tuple[float, float]]] = {
    "in_distribution": {},
    "fast_pll": {"pll_hz": (50.0, 100.0)},
    "fast_dc_loop": {"dc_hz": (30.0, 60.0)},
    "very_weak": {"scr": (1.2, 1.5)},
}

COLUMNS = (
    "solved",
    "sigma",  # largest real part of any mode, 1/s
    "f",  # its frequency in the synchronous frame, Hz
    "v_t",  # terminal voltage magnitude, pu
    "iq",  # reactive current, pu
    "poles",  # unstable poles of the converter alone on an ideal voltage
    "count",  # unstable modes by the Nyquist count from the 2 x 2 scans
    "count_pos",  # the same from positive-sequence scans only
    "count_rated",  # 2 x 2 scans, admittance taken at zero reactive current
    "small_gain",  # largest loop gain where the converter is not passive
    "f_passive",  # highest frequency at which the converter is not passive, Hz
    "g_min",  # most negative conductance of the converter, pu
    "scan_sigma",  # growth rate from the positive-sequence reactance crossing, 1/s
    "scan_f",
    "sigma_slow_pll",  # sigma with the PLL ten times slower
    "sigma_slow_dc",  # sigma with the DC voltage loop ten times slower
    "sigma_no_ff",  # sigma without voltage feed-forward
    # Scores on the eigenvalues of the loop matrix, and coarse and noisy scans (step 18).
    *adm.SCREEN_COLUMNS[2:],
)
_GRID = adm.scan_grid()
_S = 2j * np.pi * _GRID


def spec_hash() -> str:
    blob = json.dumps(
        {"version": VERSION, "seed": SEED, "spec": [vars(p) for p in SPEC], "splits": SPLITS},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sample(n: int, family: str = "pll", split: str = "in_distribution") -> np.ndarray:
    """Scrambled Sobol points scaled to the ranges of a split. Shape (n, 12). In the
    `follow` family power runs from -1 to -0.1 pu."""
    from scipy.stats import qmc

    seed = SEED + 1000 * list(SPLITS).index(split) + 100 * FAMILIES.index(family)
    unit = qmc.Sobol(d=len(SPEC), scramble=True, seed=seed).random(n)
    out = np.empty_like(unit)
    for j, par in enumerate(SPEC):
        lo, hi = SPLITS[split].get(par.name, (par.low, par.high))
        if par.name == "p" and family == "follow":
            lo, hi = -1.0, -0.1
        if par.log:
            out[:, j] = np.exp(np.log(lo) + unit[:, j] * (np.log(hi) - np.log(lo)))
        else:
            out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def network(th: dict) -> dict:
    """Network impedances from the short-circuit ratio and the other network numbers."""
    x_grid = 1.0 / th["scr"] - X_T
    x_e = x_grid * (1.0 + th["par_ratio"]) / th["par_ratio"]
    r_e = x_e / th["xr"]
    return {
        "r_e": r_e, "x_e": x_e, "x_c": th["k"] * x_e,
        "r_p": th["par_ratio"] * r_e, "x_p": th["par_ratio"] * x_e,
        "r_t": X_T / XR_PLANT, "x_t": X_T,
    }  # fmt: skip


FOLLOW_FILTER = 0.1  # a waveform-following converter measures the voltage ten times faster


def controls(th: dict, follow: float) -> ConverterControls:
    """Without a phase-locked loop the filter acts on the 60 Hz waveform itself, and a
    filter of milliseconds would put the current tens of degrees behind the voltage."""
    return ConverterControls(
        cc_hz=th["cc_hz"], pll_hz=th["pll_hz"], dc_hz=th["dc_hz"], c_dc=th["c_dc"], ff=th["ff"],
        t_ff=th["t_ff"] * (FOLLOW_FILTER if follow else 1.0), x_f=th["x_f"], follow=follow,
    )  # fmt: skip


def operating_point(
    theta: np.ndarray, follow: float, ctl: ConverterControls | None = None, net: dict | None = None
):
    """Model, steady state, parameters and network of one case. Raises ValueError if
    there is no steady state. `net` replaces the network of the case."""
    th = dict(zip(NAMES, map(float, theta), strict=True))
    net = network(th) if net is None else dict(net)
    model = Converter(ctl or controls(th, follow))
    x0, p = model.equilibrium(th["p"], **net)
    return model, x0, p, net


def growth(model: Converter, x0, p) -> tuple[float, float]:
    """Largest real part of the modes of the full model (1/s) and its frequency (Hz)."""
    keep = active_states(model.controls.follow)
    lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy()[np.ix_(keep, keep)])
    top = int(np.argmax(lam.real))
    return float(lam.real[top]), float(abs(lam.imag[top]) / (2 * np.pi))


def evaluate(theta: np.ndarray, follow: float) -> np.ndarray:
    """One case, in the order of COLUMNS. NaN and solved = 0 without a steady state."""
    out = np.full(len(COLUMNS), np.nan)
    out[0] = 0.0
    try:
        model, x0, p, net = operating_point(theta, follow)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return out
    sigma, f = growth(model, x0, p)
    vt = model.terminal_voltage(x0, p)
    v_t = complex(float(vt[0]), float(vt[1]))
    i = complex(float(x0[0] + x0[2]), float(x0[1] + x0[3]))

    scan = adm.ConverterScan(model, v_t, i, p)
    y, z = scan.admittance(_S), adm.network_impedance(net, _S)
    index = adm.passivity_index(y)
    bad = index < 0
    scan_sigma, scan_f = adm.scan_growth(y, z, _GRID)

    if follow:
        count_rated = adm.unstable_modes(y, z, scan.unstable_poles)
    else:
        rated = adm.ConverterScan(
            model, v_t, model.current_at(v_t, float(theta[0])), {**p, "iq_ref": _t(0.0)}
        )
        count_rated = adm.unstable_modes(rated.admittance(_S), z, rated.unstable_poles)

    base = model.controls
    slowed = []
    for change in ({"pll_hz": base.pll_hz / 10}, {"dc_hz": base.dc_hz / 10}, {"ff": 0.0}):
        try:
            m2, x2, p2, _ = operating_point(theta, follow, replace(base, **change))
            slowed.append(growth(m2, x2, p2)[0])
        except ValueError:
            slowed.append(np.nan)

    out[:] = [
        1.0, sigma, f, abs(v_t), float(p["iq_ref"]), scan.unstable_poles,
        adm.unstable_modes(y, z, scan.unstable_poles),
        adm.unstable_modes(y, z, scan.unstable_poles, coupled=False),
        count_rated,
        adm.small_gain_score(y, z),
        float(np.abs(_GRID[bad]).max()) if bad.any() else 0.0,
        float(index.min()),
        scan_sigma, scan_f,
        *slowed,
        *adm.screen_scores(y, z, scan.unstable_poles, case_seed(theta))[2:],
    ]  # fmt: skip
    return out


def scan_pair(theta: np.ndarray, follow: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Converter admittance and network impedance of one case on `admittance.scan_grid()`,
    and the number of unstable poles of the converter alone."""
    model, x0, p, net = operating_point(theta, follow)
    vt = model.terminal_voltage(x0, p)
    v_t = complex(float(vt[0]), float(vt[1]))
    i = complex(float(x0[0] + x0[2]), float(x0[1] + x0[3]))
    scan = adm.ConverterScan(model, v_t, i, p)
    return scan.admittance(_S), adm.network_impedance(net, _S), scan.unstable_poles


def _fit(args) -> np.ndarray:
    theta, follow = args
    try:
        y, z, _ = scan_pair(theta, follow)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.full(len(adm.FIT_COLUMNS), np.nan)
    return adm.fit_scores(y, z, _GRID, case_seed(theta))


def fits(theta: np.ndarray, family: str, processes: int = 1) -> np.ndarray:
    """Growth rate from rational fits to the scans of every row of theta, in the order
    of `admittance.FIT_COLUMNS`. Row order is kept."""
    jobs = [(row, float(family == "follow")) for row in theta]
    if processes <= 1:
        return np.array([_fit(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(_fit, jobs, chunksize=64))


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _job(args) -> np.ndarray:
    return evaluate(*args)


def generate(theta: np.ndarray, family: str, processes: int = 1) -> np.ndarray:
    """Evaluate every row of theta. Row order is kept."""
    follow = float(family == "follow")
    jobs = [(row, follow) for row in theta]
    if processes <= 1:
        return np.array([_job(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(_job, jobs, chunksize=64))


def folds(n: int) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, in fixed proportions 80 / 10 / 10."""
    order = np.random.default_rng(SEED).permutation(n)
    fold = np.zeros(n, dtype=np.int8)
    fold[order[int(0.8 * n) : int(0.9 * n)]] = 1
    fold[order[int(0.9 * n) :]] = 2
    return fold
