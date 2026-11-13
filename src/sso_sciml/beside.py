"""A doubly-fed plant beside a converter, on one bus and with impedance between them.

The doubly-fed plant is System B2 with its full controls (dataset T1's machine and
control parameters). The converter is the generic one of step 17 (dataset C1), a source
or a load. They share the network of the converter's case. Two layouts:

    bus    both at one terminal, one transformer of X_T to the plant bus (as step 27)
    apart  each behind its own transformer of X_T on its own rating, meeting at the
           plant bus

On the common rating the two transformers in parallel are again X_T, so the network seen
from far away is the same in both. What differs is whether the units see the same
voltage.

The doubly-fed plant holds zero reactive power at its terminal. A converter with a
phase-locked loop holds its own terminal voltage at 1 pu; one without follows the
voltage waveform.

Each unit is linearised at the joint steady state and the units are closed on the
network exactly (`modes`). Scored against that:

    alone      each unit by itself on the network
    added      each unit scanned by itself on the network, the scans set side by side
    reference  the screen of step 24: each unit scanned at the joint steady state with
               1 % noise, fitted, closed on a network within 10 %, as given and at the
               32 ends of its ranges

Used by scripts/machine_beside_converter.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import eig
from scipy.optimize import fsolve

from . import admittance as adm
from . import vectorfit as vf
from .data import c1, t1
from .neterror import FACTORS, perturb
from .physics.converter import WB, Converter
from .physics.dfig_full import Controls, DFIGFull
from .physics.induction import IGData
from .scanfit import BANDS, band_growth, noisy

J = vf.J
KINDS = ("pll", "follow")  # family of the converter
LAYOUTS = ("bus", "apart")
Z_OWN = complex(c1.X_T / c1.XR_PLANT, c1.X_T)  # a unit's transformer, on its own rating
NOISE, BOUND = 0.01, 0.10  # the reference screen of step 24
_PER = ("top", *BANDS, "f_top")
COLUMNS = (
    "share",  # of the doubly-fed plant in the common rating
    "p_machine", "p_converter",  # power out of each on its own rating, pu
    "w_r",
    *(f"{name}_{lay}" for lay in LAYOUTS
      for name in ("solved", *_PER, "v_machine", "iq", "poles",
                   *(f"given_{b}" for b in BANDS), *(f"ends_{b}" for b in BANDS))),
    *(f"{name}_machine" for name in ("solved", *_PER)),
    *(f"{name}_converter" for name in ("solved", *_PER)),
    *(f"{name}_added" for name in _PER),
)  # fmt: skip
_GRID = adm.scan_grid()
_F = _GRID[_GRID >= 0][:: adm.NOISE_STEP]
_SIGNS = np.array(np.meshgrid(*[(-1.0, 1.0)] * len(FACTORS))).reshape(len(FACTORS), -1).T
_ERRORS = (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError,
           np.linalg.LinAlgError)  # fmt: skip


@dataclass(frozen=True)
class Machine:
    """The doubly-fed plant of one row of dataset T1's parameters. Only the machine and
    its controls are taken from the row, and the rotor speed; the network is not."""

    model: DFIGFull
    w_r: float

    @property
    def power(self) -> float:
        return float(self.w_r**3 / self.model.controls.w_rated**3)

    def scan(self, v: complex) -> adm.Scan:
        """The plant alone at terminal voltage v, in the frame v is given in."""
        x0, p = self.model.equilibrium(0.0, w_r=self.w_r, v_b=abs(v), r_e=0.0, x_e=0.0)
        own = adm.DFIGScan(self.model, x0, p)  # in the frame of its terminal voltage
        if own.residual > 1e-8:
            raise ValueError("steady state not found")
        cos, sin = v.real / abs(v), v.imag / abs(v)
        rot = np.array([[cos, -sin], [sin, cos]])
        return adm.Scan(own.a, own.b @ rot.T, rot @ own.c)


def machine(theta: np.ndarray) -> Machine:
    th = dict(zip(t1.NAMES, map(float, theta), strict=True))
    ctl = Controls(rsc_hz=th["rsc_hz"], pll_hz=th["pll_hz"], ff=th["ff"],
                   outer_hz=th["outer_ratio"] * th["rsc_hz"])  # fmt: skip
    return Machine(DFIGFull(data=replace(IGData(), r_r=th["r_r"]), controls=ctl), th["w_r"])


@dataclass(frozen=True)
class Unit:
    """One unit as a state-space model at its steady state: dx/dt = A x + B v, current
    drawn = C x + D v on its own rating. `z` is the impedance between its terminal and
    the junction, on its own rating."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    share: float
    z: complex = 0.0


def _unit(model, share: float, z: complex) -> Unit:
    if isinstance(model, dict):
        return Unit(model["a"], model["b"], model["c"], model["d"], share, z)
    return Unit(model.a, model.b, model.c, np.zeros((2, 2)), share, z)


def modes(units: list[Unit], net: dict) -> np.ndarray:
    """Modes of the units on the network of `admittance.network_impedance`. Each unit
    reaches a junction through its own impedance; r_t and x_t of the network lie between
    the junction and the plant bus. With every own impedance zero this is
    `vectorfit.closed_loop_modes` on the units side by side.

    Unknowns: unit states, unit terminal voltages, line current, path current, capacitor
    voltage, plant-bus voltage, junction voltage.
    """
    parts = [vf.balance(u.a, u.b, u.c) for u in units]
    sizes = [p[0].shape[0] for p in parts]
    n, k = sum(sizes), len(units)
    cap = net["x_c"] != 0
    start = np.concatenate([[0], np.cumsum(sizes)])
    xs = [slice(start[j], start[j + 1]) for j in range(k)]
    vs = [slice(n + 2 * j, n + 2 * j + 2) for j in range(k)]
    m = n + 2 * k
    ic, ip, vc = slice(m, m + 2), slice(m + 2, m + 4), slice(m + 4, m + 6)
    vm = slice(m + 4 + 2 * cap, m + 6 + 2 * cap)
    vj = slice(m + 6 + 2 * cap, m + 8 + 2 * cap)
    size = m + 8 + 2 * cap
    e, f = np.zeros((size, size), dtype=complex), np.zeros((size, size), dtype=complex)
    eye = np.eye(2)
    for u, (a, b, c), x, v in zip(units, parts, xs, vs, strict=True):
        e[x, x] = np.eye(a.shape[0])
        f[x, x], f[x, v] = a, b
        # Own impedance on the common rating, carrying the current the unit draws:
        # (x / wb) di/dt = v_junction - v_unit - (r + x J) i, with i = share (C x + D v).
        z = u.z / u.share
        z_mat = z.real * eye + z.imag * J
        e[v, x], e[v, v] = z.imag / WB * u.share * c, z.imag / WB * u.share * u.d
        f[v, vj], f[v, v], f[v, x] = eye, -eye - z_mat @ (u.share * u.d), -z_mat @ (u.share * c)
        # The currents out of the units leave through the two branches.
        f[vm, x], f[vm, v] = u.share * c, u.share * u.d
    f[vm, ic] = f[vm, ip] = eye
    e[ic, ic] = net["x_e"] / WB * eye
    f[ic, ic], f[ic, vm] = -(net["r_e"] * eye + net["x_e"] * J), eye
    e[ip, ip] = net["x_p"] / WB * eye
    f[ip, ip], f[ip, vm] = -(net["r_p"] * eye + net["x_p"] * J), eye
    if cap:
        f[ic, vc] = -eye
        e[vc, vc] = eye
        f[vc, ic], f[vc, vc] = WB * net["x_c"] * eye, -WB * J
    e[vj, ic] = e[vj, ip] = net["x_t"] / WB * eye
    f[vj, vj], f[vj, vm] = eye, -eye
    f[vj, ic] = f[vj, ip] = -(net["r_t"] * eye + net["x_t"] * J)
    alpha, beta = eig(f, e, right=False, homogeneous_eigvals=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = alpha / beta
    return lam[np.isfinite(lam) & (np.abs(lam) < vf.MODE_LIMIT)]


def layout(net: dict, name: str) -> tuple[dict, complex]:
    """The network and the units' own impedance in a layout."""
    if name == "bus":
        return dict(net), 0.0
    return {**net, "r_t": 0.0, "x_t": 0.0}, Z_OWN


def operating_point(
    mach: Machine | None,
    conv: tuple[Converter, float] | None,
    share: float,
    net: dict,
    z_own: complex,
) -> dict:
    """Joint steady state. `share` is the doubly-fed plant's part of the common rating;
    a unit that is absent leaves its part empty. Returns the terminal voltage and the
    current out of each unit (own rating) and the converter's reactive current."""
    line = net["r_e"] + 1j * (net["x_e"] - net["x_c"])
    path = net["r_p"] + 1j * net["x_p"]
    z_g = net["r_t"] + 1j * net["x_t"] + line * path / (line + path)
    follow = bool(conv[0].controls.follow) if conv else False

    def unpack(w):
        v_j, k, out = complex(w[0], w[1]), 2, {}
        if mach:
            i = complex(w[k], w[k + 1])
            out["v_machine"], out["i_machine"] = v_j + z_own * i, i
            k += 2
        if conv:
            model, p = conv
            if follow:
                v = complex(w[k], w[k + 1])
                out["iq"], i = 0.0, model.current_at(v, p)
            else:
                v, out["iq"] = np.exp(1j * w[k]), float(w[k + 1])
                i = model.current_at(v, p, out["iq"])
            out["v_converter"], out["i_converter"] = v, i
        return v_j, out

    def mismatch(w):
        v_j, o = unpack(w)
        total, res = 0.0, []
        if mach:
            s = o["v_machine"] * np.conj(o["i_machine"]) - mach.power
            total += share * o["i_machine"]
            res += [s.real, s.imag]
        if conv:
            r = o["v_converter"] - v_j - z_own * o["i_converter"]
            total += (1.0 - share) * o["i_converter"]
            res += [r.real, r.imag]
        r = v_j - 1.0 - z_g * total
        return [r.real, r.imag, *res]

    start = [1.0, 0.0]
    if mach:
        start += [mach.power, 0.0]
    if conv:
        start += [1.0, 0.0] if follow else [0.0, 0.0]
    with np.errstate(all="ignore"):
        w = fsolve(mismatch, start, xtol=1e-12, full_output=True)[0]
        ok = np.all(np.isfinite(w)) and np.max(np.abs(mismatch(w))) < 1e-9
    if not ok:
        raise ValueError("steady state not found")
    return unpack(w)[1]


def scans(mach, conv, share: float, net: dict, z_own: complex) -> tuple[list, list, dict]:
    """Each unit alone at the joint steady state, their shares, and that steady state."""
    op = operating_point(mach, conv, share, net, z_own)
    out, shares = [], []
    if mach:
        out.append(mach.scan(op["v_machine"]))
        shares.append(share)
    if conv:
        model, p = conv
        out.append(adm.ConverterScan(model, op["v_converter"], op["i_converter"],
                                     model.params(p, iq_ref=op["iq"])))  # fmt: skip
        shares.append(1.0 - share)
    return out, shares, op


def _summary(lam: np.ndarray) -> list[float]:
    top = int(np.argmax(lam.real))
    return [lam.real[top], *band_growth(lam), abs(lam.imag[top]) / (2 * np.pi)]


def _together(models: list, shares: list[float], net: dict, z_own: complex) -> np.ndarray:
    return modes([_unit(m, w, z_own) for m, w in zip(models, shares, strict=True)], net)


def reference(models: list, shares: list[float], net: dict, z_own: complex,
              rng: np.random.Generator) -> list[float]:  # fmt: skip
    """The screen of step 24 on several units: growth rate in each band on the network as
    given, then the largest over the 32 ends of its ranges. The fifth network quantity,
    the plant-side impedance, is the units' own transformers in the `apart` layout."""
    s = 2j * np.pi * _F
    fits = [vf.fit_admittance(_F, noisy(m.admittance(s), NOISE, rng), NOISE) for m in models]
    width = np.log1p(BOUND)
    factor = np.exp(rng.uniform(-width, width, size=len(FACTORS)))

    def bands(f: np.ndarray) -> list[float]:
        return band_growth(_together(fits, shares, perturb(net, f), z_own * f[4]))

    ends = np.max([bands(factor * np.exp(width * sign)) for sign in _SIGNS], axis=0)
    return [*bands(factor), *ends]


def case(args: tuple[str, np.ndarray, np.ndarray, float]) -> np.ndarray:
    """One case in the order of COLUMNS: a row of dataset T1's parameters for the
    doubly-fed plant, a row of dataset C1's for the converter and the network, and the
    doubly-fed plant's share of the common rating."""
    kind, theta_m, theta_c, share = args
    thc = dict(zip(c1.NAMES, map(float, theta_c), strict=True))
    net = c1.network(thc)
    mach = machine(theta_m)
    conv = (Converter(c1.controls(thc, float(kind == "follow"))), thc["p"])
    rng = np.random.default_rng(t1.case_seed(np.concatenate([theta_m, theta_c])) + 32)
    out = [share, mach.power, thc["p"], mach.w_r]
    width = 1 + len(_PER) + 3 + 2 * len(BANDS)
    for name in LAYOUTS:
        lay_net, z_own = layout(net, name)
        try:
            models, shares, op = scans(mach, conv, share, lay_net, z_own)
            lam = _together(models, shares, lay_net, z_own)
            out += [1.0, *_summary(lam), abs(op["v_machine"]), op["iq"],
                    sum(m.unstable_poles for m in models),
                    *reference(models, shares, lay_net, z_own, rng)]  # fmt: skip
        except _ERRORS:
            out += [0.0] + [np.nan] * (width - 1)
    # Each unit by itself on the network, at one terminal.
    alone = []
    for pair in ((mach, None), (None, conv)):
        try:
            models, shares, _ = scans(*pair, share, net, 0.0)
            alone.append(models[0])
            out += [1.0, *_summary(_together(models, shares, net, 0.0))]
        except _ERRORS:
            alone.append(None)
            out += [0.0] + [np.nan] * len(_PER)
    if all(m is not None for m in alone):
        out += _summary(_together(alone, [share, 1.0 - share], net, 0.0))
    else:
        out += [np.nan] * len(_PER)
    return np.array(out, dtype=float)


def sample(n: int, kind: str) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """n cases: a row of dataset T1's parameter space, an independent row of dataset
    C1's, and a share between 0.2 and 0.8. The same (n, kind) always returns the same
    cases."""
    rng = np.random.default_rng(c1.SEED + 32 + KINDS.index(kind))
    theta_m = t1.sample(n)[rng.permutation(n)]
    theta_c = c1.sample(n, kind)
    shares = rng.uniform(0.2, 0.8, size=n)
    return [(kind, m, c, float(a)) for m, c, a in zip(theta_m, theta_c, shares, strict=True)]


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def cases(jobs: list, processes: int = 1) -> np.ndarray:
    """`case` for every job. Order is kept."""
    if processes <= 1:
        return np.array([case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, jobs, chunksize=16))
