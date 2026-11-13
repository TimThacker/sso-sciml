"""A network with shunt capacitance (lifts the last network limit of step 28).

The network of every step so far is R, X and a series capacitor. A real line has charging
capacitance and a collector system has cables. Here a lossless shunt capacitor of
susceptance b_sh sits on the plant bus, between the transformer and the two paths to the
infinite bus. With the inductances around it, it makes a parallel resonance of the network
above 58 Hz.

Each in-range case of dataset C1 is run with the four converter variants of step 28 (as
it is, with a control delay, with a filter capacitor, with both; the same draws), on the
network without and with the shunt capacitor. For the two variants with a delay, the
modes are also computed as a study would that

    ignore   leaves the shunt capacitance out of the network model
    given    has it wrong by a factor within BOUND either way
    ends     has it wrong by that factor and takes the worse of the two ends of its range
    fit      has it exact and uses the rational fit of step 19 on a scan to 10 kHz

Used by scripts/shunt_capacitance.py.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import eig
from scipy.optimize import fsolve

from . import admittance as adm
from . import hf
from . import vectorfit as vf
from .data import c1
from .data.t1 import case_seed
from .physics.converter import WB, Converter

J = vf.J
B_SH = (0.01, 0.30)  # shunt susceptance on the plant bus, pu, even on a logarithmic scale
BOUND = 0.20  # the susceptance is known to within this, either way
SCREENED = ("delay", "both")
_PER = ("top", "f_top", *hf.BANDS)
COLUMNS = (
    "t_d", "b_c", "r_d", "b_sh", "f_par", "f_lcl",
    *(f"{name}_{v}" for v in hf.VARIANTS
      for name in ("solved", *_PER, *(f"{n}_sh" for n in _PER), "alone")),
    *(f"{name}_{v}" for v in SCREENED for name in ("ignore", "given", "ends", "fit")),
)  # fmt: skip
_GRID = adm.scan_grid(f_max=10000.0)[:: adm.NOISE_STEP]


def network_impedance(net: dict, s: np.ndarray, b_sh: float) -> np.ndarray:
    """Z(s) seen from the converter terminal with the shunt capacitor on the plant bus,
    shape (len(s), 2, 2). As `admittance.network_impedance` for b_sh = 0."""
    s = np.asarray(s, dtype=complex)

    def z(sig):
        line = net["r_e"] + sig * net["x_e"] + net["x_c"] / sig
        path = net["r_p"] + sig * net["x_p"]
        return net["r_t"] + sig * net["x_t"] + 1.0 / (1.0 / line + 1.0 / path + sig * b_sh)

    plus, minus = z(s / WB + 1j), z(s / WB - 1j)
    z_d, z_q = 0.5 * (plus + minus), (plus - minus) / 2j
    return np.stack([np.stack([z_d, -z_q], -1), np.stack([z_q, z_d], -1)], -2)


def closed_loop_modes(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, net: dict, b_sh: float
) -> np.ndarray:
    """Modes of a plant dx/dt = A x + B v, current drawn = C x + D v, on the network with
    the shunt capacitor. As `vectorfit.closed_loop_modes` for b_sh = 0.

    Unknowns: plant state, line current, path current, series capacitor voltage,
    plant-bus voltage, terminal voltage. The plant-bus voltage is a state when b_sh > 0:
    (b_sh / wb) dv/dt = -(i_line + i_path) - i_plant - b_sh J v.
    """
    a, b, c = vf.balance(a, b, c)
    n = a.shape[0]
    cap = net["x_c"] != 0
    ic, ip = slice(n, n + 2), slice(n + 2, n + 4)
    vc = slice(n + 4, n + 6)
    vm = slice(n + 4 + 2 * cap, n + 6 + 2 * cap)
    vt = slice(n + 6 + 2 * cap, n + 8 + 2 * cap)
    size = n + 8 + 2 * cap
    e, f = np.zeros((size, size), dtype=complex), np.zeros((size, size), dtype=complex)
    eye = np.eye(2)
    e[:n, :n] = np.eye(n)
    f[:n, :n], f[:n, vt] = a, b
    e[ic, ic] = net["x_e"] / WB * eye
    f[ic, ic], f[ic, vm] = -(net["r_e"] * eye + net["x_e"] * J), eye
    e[ip, ip] = net["x_p"] / WB * eye
    f[ip, ip], f[ip, vm] = -(net["r_p"] * eye + net["x_p"] * J), eye
    if cap:
        f[ic, vc] = -eye
        e[vc, vc] = eye
        f[vc, ic], f[vc, vc] = WB * net["x_c"] * eye, -WB * J
    # Current balance on the plant bus, with the shunt capacitor.
    e[vm, vm] = b_sh / WB * eye
    f[vm, ic], f[vm, ip], f[vm, :n], f[vm, vt] = -eye, -eye, -c, -d
    f[vm, vm] = -b_sh * J
    # Plant-side impedance carries the plant current C x + D v_t:
    # (x_t / wb) d(i_plant)/dt = v_m - v_t - (r_t + x_t J) i_plant.
    z_t = net["r_t"] * eye + net["x_t"] * J
    e[vt, :n], e[vt, vt] = net["x_t"] / WB * c, net["x_t"] / WB * d
    f[vt, vm], f[vt, vt], f[vt, :n] = eye, -eye - z_t @ d, -z_t @ c
    alpha, beta = eig(f, e, right=False, homogeneous_eigvals=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = alpha / beta
    return lam[np.isfinite(lam) & (np.abs(lam) < vf.MODE_LIMIT)]


def operating_point(model: Converter, p_src: float, net: dict, b_c: float, r_d: float,
                    b_sh: float):  # fmt: skip
    """As `hf.operating_point` with the shunt capacitor on the plant bus."""
    y_f = 1.0 / (r_d - 1j / b_c) if b_c > 0 else 0.0
    line = net["r_e"] + 1j * (net["x_e"] - net["x_c"])
    path = net["r_p"] + 1j * net["x_p"]
    y_bus = 1.0 / line + 1.0 / path + 1j * b_sh
    v_th = (1.0 / line + 1.0 / path) / y_bus  # open-circuit voltage of the plant bus
    z_th = net["r_t"] + 1j * net["x_t"] + 1.0 / y_bus
    follow = bool(model.controls.follow)

    def unpack(w):
        if follow:
            v = complex(w[0], w[1])
            return v, model.current_at(v, p_src), 0.0
        v = np.exp(1j * w[0])
        return v, model.current_at(v, p_src, float(w[1])), float(w[1])

    def mismatch(w):
        v, i, _ = unpack(w)
        r = v - v_th - z_th * (i - y_f * v)
        return [r.real, r.imag]

    with np.errstate(all="ignore"):
        w = fsolve(mismatch, [1.0, 0.0] if follow else [0.0, 0.0], xtol=1e-12,
                   full_output=True)[0]  # fmt: skip
        ok = np.all(np.isfinite(w)) and np.max(np.abs(mismatch(w))) < 1e-9
    if not ok:
        raise ValueError("steady state not found")
    return unpack(w)


def solve(theta: np.ndarray, follow: float, t_d: float, b_c: float, r_d: float, b_sh: float):
    """Plant and network of one case; any of delay, filter capacitor and shunt may be 0."""
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    net = c1.network(th)
    controls = c1.controls(th, follow)
    model = hf.DelayedConverter(controls, t_d=t_d) if t_d > 0 else Converter(controls)
    v, i, iq = operating_point(model, th["p"], net, b_c, r_d, b_sh)
    return hf.PlantScan(model, v, i, model.params(th["p"], iq_ref=iq), b_c, r_d), net


def parallel_frequency(theta: np.ndarray, b_sh: float) -> float:
    """Resonance of the shunt capacitor with the inductances on the plant bus (line and
    parallel path one way, transformer and filter reactor the other), Hz in the phases.
    Series and filter capacitors left out."""
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    net = c1.network(th)
    b_l = 1.0 / net["x_e"] + 1.0 / net["x_p"] + 1.0 / (net["x_t"] + th["x_f"])
    return float(60.0 * np.sqrt(b_l / b_sh))


def draws(theta: np.ndarray) -> tuple[float, float, float, float, float]:
    """Delay, filter capacitor and damping resistor as step 28 drew them, then the shunt
    susceptance and the factor by which a study has it wrong."""
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    rng = np.random.default_rng(case_seed(theta) + 28)
    rho, b_c, r_d = (float(np.exp(rng.uniform(*np.log(hf.RANGES[k])))) for k in hf.RANGES)
    rng = np.random.default_rng(case_seed(theta) + 30)
    b_sh = float(np.exp(rng.uniform(*np.log(B_SH))))
    factor = float(np.exp(rng.uniform(-np.log1p(BOUND), np.log1p(BOUND))))
    return rho / th["cc_hz"], b_c, r_d, b_sh, factor


def _summary(lam: np.ndarray) -> list[float]:
    top = int(np.argmax(lam.real))
    return [lam.real[top], abs(lam.imag[top]) / (2 * np.pi), *hf.banded(lam)]


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS."""
    kind, theta = args
    follow = float(kind == "follow")
    t_d, b_c, r_d, b_sh, factor = draws(theta)
    out = [t_d, b_c, r_d, b_sh, parallel_frequency(theta, b_sh), hf.lcl_frequency(theta, b_c)]
    screens: dict[str, list[float]] = {}
    width = 2 * len(_PER) + 1
    for variant in hf.VARIANTS:
        delay = t_d if variant in ("delay", "both") else 0.0
        cap = b_c if variant in ("capacitor", "both") else 0.0
        try:
            plain, net = solve(theta, follow, delay, cap, r_d, 0.0)
            plant, _ = solve(theta, follow, delay, cap, r_d, b_sh)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            out += [0.0] + [np.nan] * width
            continue

        def modes(b: float, plant=plant, net=net) -> np.ndarray:
            return closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net, b)

        out += [
            1.0,
            *_summary(closed_loop_modes(plain.a, plain.b, plain.c, plain.d, net, 0.0)),
            *_summary(modes(b_sh)),
            plant.poles.real.max(),
        ]
        if variant in SCREENED:
            given = b_sh * factor
            s = 2j * np.pi * _GRID
            fit = adm.fit_growth(plant.admittance(s), network_impedance(net, s, b_sh), _GRID)[0]
            screens[variant] = [
                modes(0.0).real.max(),
                modes(given).real.max(),
                max(modes(given * (1 + BOUND) ** k).real.max() for k in (-1, 1)),
                fit,
            ]
    for variant in SCREENED:
        out += screens.get(variant, [np.nan] * 4)
    return np.array(out, dtype=float)


def _init_worker() -> None:
    torch.set_num_threads(1)


def cases(kind: str, theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`case` for every row of theta. Row order is kept."""
    jobs = [(kind, row) for row in theta]
    if processes <= 1:
        return np.array([case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, jobs, chunksize=32))
